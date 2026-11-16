# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test.statedb.protocol_common import CaseBuilder, MapLikeCase

from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.types import TaskType
from kumoapi.jobs import S3PredictionTableSource
from kumoapi.model_plan import PredictionTableGenerationPlan
from kumoapi.source_table import S3RelativeSource

from rest.backends.ml_backend.job_models import (
    GeneratePredTableJobConfig,
    GeneratePredTableJobDBEntry,
    GeneratePredTableJobOutput,
    GenerateTrainTableJobConfig,
    GenerateTrainTableJobDBEntry,
    GenerateTrainTableJobOutput,
    TrainingJobDBEntry,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
)
from rest.config.training_job import TrainingJobOutput
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobID,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import JobTags, JobTagsMap
from statedb.config.timestamp import get_current_timestamps
from statedb.ml.jobs import BatchPredictionJobDBEntry

_DEFAULT_PROJECT_ID = ProjectID("default")


def _running_training_output() -> TrainingJobOutput:
    return TrainingJobOutput(
        status=JobStatus.RUNNING,
        log=JobEventLogEntry.configuring_prediction_job(),
    )


def _running_batch_prediction_output() -> BatchPredictionJobOutput:
    return BatchPredictionJobOutput(
        status=JobStatus.RUNNING,
        log=JobEventLogEntry.configuring_prediction_job(),
    )


def _running_generate_train_table_output() -> GenerateTrainTableJobOutput:
    return GenerateTrainTableJobOutput(
        status=JobStatus.RUNNING,
        log=JobEventLogEntry.configuring_prediction_job(),
    )


def _running_generate_pred_table_output() -> GeneratePredTableJobOutput:
    return GeneratePredTableJobOutput(
        status=JobStatus.RUNNING,
        log=JobEventLogEntry.configuring_prediction_job(),
    )


def case_training_job() -> MapLikeCase[TrainingJobID, TrainingJobDBEntry]:
    return MapLikeCase(
        key=TrainingJobID.make_random_id(),
        value=TrainingJobDBEntry(
            id=TrainingJobID.make_random_id(),
            project_id=_DEFAULT_PROJECT_ID,
            execution=_running_training_output(),
            timestamps=get_current_timestamps(),
        ),
        key_type=TrainingJobID,
        value_type=TrainingJobDBEntry,
    )


def case_batch_prediction_job(
) -> MapLikeCase[BatchPredictionJobID, BatchPredictionJobDBEntry]:
    return MapLikeCase(
        key=BatchPredictionJobID.make_random_id(),
        value=BatchPredictionJobDBEntry(
            id=BatchPredictionJobID.make_random_id(),
            project_id=_DEFAULT_PROJECT_ID,
            config=BatchPredictionJobConfig(
                predictive_query_id=PredictiveQueryID("p"),
                training_job_id=TrainingJobID("trainingjob-1"),
                graph_id="graph_test",
                custom_pred_table_source=S3PredictionTableSource(
                    connector_id="connector_00000000000000000000000000000001",
                    source=S3RelativeSource(path="predictions.parquet"),
                ),
                task_type=TaskType.REGRESSION,
            ),
            execution=_running_batch_prediction_output(),
            timestamps=get_current_timestamps(),
        ),
        key_type=BatchPredictionJobID,
        value_type=BatchPredictionJobDBEntry,
    )


def case_generate_train_table_job(
) -> MapLikeCase[GenerateTrainTableJobID, GenerateTrainTableJobDBEntry]:
    return MapLikeCase(
        key=GenerateTrainTableJobID.make_random_id(),
        value=GenerateTrainTableJobDBEntry(
            id=GenerateTrainTableJobID.make_random_id(),
            project_id=_DEFAULT_PROJECT_ID,
            config=GenerateTrainTableJobConfig(
                pquery_id=PredictiveQueryID("p"),
                plan=TrainingTableGenerationPlan(),
                graph_id="graph-test",
            ),
            execution=_running_generate_train_table_output(),
            timestamps=get_current_timestamps(),
        ),
        key_type=GenerateTrainTableJobID,
        value_type=GenerateTrainTableJobDBEntry,
    )


def case_generate_pred_table_job(
) -> MapLikeCase[GeneratePredictionTableJobID, GeneratePredTableJobDBEntry]:
    return MapLikeCase(
        key=GeneratePredictionTableJobID.make_random_id(),
        value=GeneratePredTableJobDBEntry(
            id=GeneratePredictionTableJobID.make_random_id(),
            project_id=_DEFAULT_PROJECT_ID,
            config=GeneratePredTableJobConfig(
                pquery_id=PredictiveQueryID("p"),
                plan=PredictionTableGenerationPlan(),
                graph_id="graph-test",
            ),
            execution=_running_generate_pred_table_output(),
            timestamps=get_current_timestamps(),
        ),
        key_type=GeneratePredictionTableJobID,
        value_type=GeneratePredTableJobDBEntry,
    )


def case_job_tags() -> MapLikeCase[JobType, JobTagsMap]:
    return MapLikeCase(
        key=JobType.TRAINING_JOB,
        value=JobTagsMap(
            mapping={
                JobID("job-1"): JobTags(timestamps=get_current_timestamps())
            }),
        key_type=JobType,
        value_type=JobTagsMap,
    )


ML_PROTOCOL_CASE_BUILDERS: list[tuple[str, CaseBuilder]] = [
    ("ml_training_job", case_training_job),
    ("ml_batch_prediction_job", case_batch_prediction_job),
    ("ml_generate_train_table_job", case_generate_train_table_job),
    ("ml_generate_pred_table_job", case_generate_pred_table_job),
    ("ml_job_tags", case_job_tags),
]
