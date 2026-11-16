# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types import PredictionArtifactType

from rest.config.mlops import (
    JobRegistryQuery,
    MLOpsQueryConfig,
    MLOpsStats,
    MLOpsStatsQuery,
    MLOpsStatType,
)
from rest.config.prediction_output import S3PredictionOutput
from rest.driver.mlops_logic import MLOpsLogic
from statedb.config.common import BatchPredictionJobID


@pytest.mark.asyncio
async def test_prediction_outputs_use_state_service_rows_without_legacy_scan():
    bp_id = BatchPredictionJobID("bp-output")
    output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id="connector_0123456789abcdef0123456789abcdef",
        file_path="s3://bucket/predictions/",
    )
    ml_job_service = SimpleNamespace(get_batch_prediction_rows=AsyncMock(
        return_value=[
            SimpleNamespace(
                id=bp_id,
                config=SimpleNamespace(outputs=[output]),
            )
        ]))
    ml_job_backend = SimpleNamespace()
    stats = MLOpsStats(
        job_status_dict={bp_id: "done"},
        bp_stats={bp_id: {}},
    )
    query = MLOpsQueryConfig(
        job_query=JobRegistryQuery(),
        stat_query=MLOpsStatsQuery(
            stat_request=[MLOpsStatType.PREDICTION_OUTPUTS]),
    )

    result = await MLOpsLogic(
        ml_job_backend,
        MagicMock(),
        state_services=SimpleNamespace(ml_jobs=ml_job_service),
    )._add_rest_db_stats(stats, query, MagicMock())

    assert result.bp_stats[bp_id][MLOpsStatType.PREDICTION_OUTPUTS] == [output]
    ml_job_service.get_batch_prediction_rows.assert_awaited_once_with(
        {bp_id}, project_ids=None)
