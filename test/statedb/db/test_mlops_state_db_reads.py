# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.mlops import (
    JobRegistryQuery,
    MLOpsQueryConfig,
    MLOpsStatsQuery,
    MLOpsStatType,
)
from statedb.config.common import JobType


class _FakeMLJobService:

    def __init__(self):
        self.training_job = SimpleNamespace(
            job_id="trainingjob-abc",
            job_type=JobType.TRAINING_JOB,
            status="DONE",
            input_data_stats={"graph_size_bytes": 11},
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            kumo_version="train-version",
            trained_job_id=None,
        )
        self.bp_job = SimpleNamespace(
            job_id="batchpredictionjob-def",
            job_type=JobType.BATCH_PREDICTION_JOB,
            status="DONE",
            input_data_stats={"graph_size_bytes": 22},
            start_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 2, 2, tzinfo=timezone.utc),
            kumo_version="pred-version",
            trained_job_id=self.training_job.job_id,
        )

    async def select_jobs(self, filters):
        if filters.get("job_type") == JobType.BATCH_PREDICTION_JOB.value:
            return [self.bp_job]
        return [self.training_job, self.bp_job]

    async def get_multi_train_jobs(self, train_job_ids):
        assert train_job_ids == [self.training_job.job_id]
        return {self.training_job.job_id: self.training_job}


@pytest.mark.asyncio
async def test_get_mlops_jobs_and_db_stats_reads_state_db_service():
    backend = MLJobBackend.__new__(MLJobBackend)
    backend._ml_job_service = _FakeMLJobService()

    stats = await backend.get_mlops_jobs_and_db_stats(
        MLOpsQueryConfig(
            job_query=JobRegistryQuery(job_filters={
                "job_type": JobType.BATCH_PREDICTION_JOB.value
            }),
            stat_query=MLOpsStatsQuery(stat_request=[
                MLOpsStatType.TABLE_STATS,
                MLOpsStatType.START_TIME,
                MLOpsStatType.END_TIME,
                MLOpsStatType.PREDICTION_VERSION,
                MLOpsStatType.TRAINING_VERSION,
                MLOpsStatType.TRAINED_JOB_ID,
            ]),
        ))

    bp_stats = stats.bp_stats["batchpredictionjob-def"]
    assert bp_stats[MLOpsStatType.TABLE_STATS] == {"graph_size_bytes": 22}
    assert bp_stats[MLOpsStatType.PREDICTION_VERSION] == "pred-version"
    assert bp_stats[MLOpsStatType.TRAINING_VERSION] == "train-version"
    assert bp_stats[MLOpsStatType.TRAINED_JOB_ID] == "trainingjob-abc"
    assert stats.train_stats == {}
