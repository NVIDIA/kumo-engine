# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from tempfile import TemporaryDirectory

import pytest
from kumo_core.types import PredictionArtifactType, TaskType
from kumoapi.jobs import (
    ADLSPredictionTableSource,
    DatabricksPredictionTableSource,
    GCSPredictionTableSource,
    S3PredictionTableSource,
)
from kumoapi.source_table import (
    ADLSRelativeSource,
    GCSRelativeSource,
    S3RelativeSource,
)
from pydantic import TypeAdapter

from io_utils.json import save_json
from kumo.stats.train_table_stats import PRED_TIME
from rest.config.batch_prediction_job import (
    BatchPredictionJobAnchorTime,
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
    BatchPredictionJobResponse,
)
from rest.config.prediction_output import (
    LocalPredictionOutput,
    PredictionStorageType,
    S3PredictionOutput,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GraphLink,
    JobEventLogEntry,
    JobStatus,
    PredictiveQueryID,
    PredictiveQueryLink,
    ProjectID,
    TrainingJobID,
)
from statedb.config.jobs import \
    S3PredictionOutput as SerializedS3PredictionOutput
from statedb.config.timestamp import ResourceTimestamps, now_utc

_CONNECTOR_ID = f'connector_{"1" * 32}'
_PROJECT_ID = ProjectID(f'project_{"1" * 32}')


@pytest.mark.parametrize(
    'source',
    [
        S3PredictionTableSource(
            connector_id=_CONNECTOR_ID,
            source=S3RelativeSource(path='predictions/table.parquet'),
        ),
        GCSPredictionTableSource(
            connector_id=_CONNECTOR_ID,
            source=GCSRelativeSource(path='predictions/table.parquet'),
        ),
        ADLSPredictionTableSource(
            connector_id=_CONNECTOR_ID,
            source=ADLSRelativeSource(path='predictions/table.parquet'),
        ),
        DatabricksPredictionTableSource(
            connector_id=_CONNECTOR_ID,
            table='prediction_input',
        ),
    ],
)
def test_batch_prediction_config_round_trips_custom_prediction_source(
        source) -> None:
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        custom_pred_table_source=source,
    )

    adapter = TypeAdapter(BatchPredictionJobConfig)
    validated = adapter.validate_python(adapter.dump_python(config))

    assert validated.custom_pred_table_source == source


def test_get_stats_from_per_job_dir_v1():
    # Test loading anchor time from job directory
    with TemporaryDirectory() as temp_dir:
        job_id = "test_job"
        pred_time = datetime(2025, 2, 11, 12, 0)

        # Create outputs dir and stats file with prediction time
        outputs_dir = os.path.join(temp_dir, "outputs")
        os.makedirs(outputs_dir)
        stats_path = os.path.join(outputs_dir, "pquery_table_stats.json")
        stats = {
            PRED_TIME: {
                "value": "kumo_timestamp_2025-02-11 12:00:00",
                "approx": False,
                "data_version": None,
                "num_entries": None,
                "timestamp": None,
            }
        }
        save_json(stats, stats_path)

        # Test loading stats
        anchor_time = BatchPredictionJobAnchorTime()
        result = anchor_time.get_stats_from_per_job_dir(job_id, temp_dir)

        assert result is not None
        assert result.anchor_time == pred_time


def test_get_stats_from_per_job_dir_missing_file():
    # Test behavior when stats file doesn't exist
    with TemporaryDirectory() as temp_dir:
        job_id = "test_job"

        anchor_time = BatchPredictionJobAnchorTime()
        result = anchor_time.get_stats_from_per_job_dir(job_id, temp_dir)

        assert result is None


def test_get_stats_from_per_job_dir_missing_pred_time():
    # Test behavior when stats file exists but has no prediction time
    with TemporaryDirectory() as temp_dir:
        job_id = "test_job"

        # Create stats file without prediction time
        outputs_dir = os.path.join(temp_dir, "outputs")
        os.makedirs(outputs_dir)
        stats_path = os.path.join(outputs_dir, "pquery_table_stats.json")
        stats = {"other_stat": {"value": 123}}
        save_json(stats, stats_path)

        anchor_time = BatchPredictionJobAnchorTime()
        result = anchor_time.get_stats_from_per_job_dir(job_id, temp_dir)

        assert result is not None
        assert result.anchor_time is None


def test_batch_prediction_response_accepts_rest_config_outputs():
    rest_output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        storage_type=PredictionStorageType.S3,
        connector_id=_CONNECTOR_ID,
        file_path="predictions.csv",
    )
    serialized_output = SerializedS3PredictionOutput(
        artifact_type=PredictionArtifactType.EMBEDDINGS,
        storage_type=PredictionStorageType.S3,
        connector_id=_CONNECTOR_ID,
        file_path="embeddings.csv",
    )
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID("pquery"),
        training_job_id=TrainingJobID("training-job"),
        task_type=TaskType.REGRESSION,
        outputs=[rest_output, serialized_output],
        graph_id="graph",
    )

    response = BatchPredictionJobResponse(
        id=BatchPredictionJobID("batch-prediction-job"),
        project_id=_PROJECT_ID,
        config=config,
        timestamps=ResourceTimestamps(
            created_at=now_utc(),
            updated_at=now_utc(),
        ),
        predictive_query_link=PredictiveQueryLink(
            id=PredictiveQueryID("pquery"),
            name=PredictiveQueryID("pquery"),
        ),
        execution=BatchPredictionJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.not_started(),
        ),
        graph_link=GraphLink(id="graph", name="graph"),
    )

    assert response.config is config
    assert all(
        isinstance(output, S3PredictionOutput)
        for output in response.config.outputs)


def test_batch_prediction_config_round_trips_local_outputs():
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.REGRESSION,
        outputs=[
            LocalPredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS),
            LocalPredictionOutput(
                artifact_type=PredictionArtifactType.EMBEDDINGS),
        ],
        graph_id='graph',
    )

    adapter = TypeAdapter(BatchPredictionJobConfig)
    validated = adapter.validate_python(adapter.dump_python(config))

    assert [output.artifact_type for output in validated.outputs] == [
        PredictionArtifactType.PREDICTIONS,
        PredictionArtifactType.EMBEDDINGS,
    ]
    assert all(
        isinstance(output, LocalPredictionOutput)
        for output in validated.outputs)


def test_multilabel_batch_prediction_config_defaults_threshold() -> None:
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
    )

    assert config.binary_classification_threshold == 0.5


def test_multilabel_batch_prediction_config_preserves_zero_threshold() -> None:
    config = BatchPredictionJobConfig(
        predictive_query_id=PredictiveQueryID('pquery'),
        training_job_id=TrainingJobID('training-job'),
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        binary_classification_threshold=0.0,
    )

    assert config.binary_classification_threshold == 0.0


@pytest.mark.parametrize('threshold', [-0.01, 1.01])
def test_multilabel_batch_prediction_config_rejects_invalid_threshold(
    threshold: float, ) -> None:
    with pytest.raises(ValueError, match='between 0 and 1'):
        BatchPredictionJobConfig(
            predictive_query_id=PredictiveQueryID('pquery'),
            training_job_id=TrainingJobID('training-job'),
            task_type=TaskType.MULTILABEL_CLASSIFICATION,
            binary_classification_threshold=threshold,
        )


def test_batch_prediction_response_model_field_accepts_rest_config_outputs():
    output = S3PredictionOutput(
        artifact_type=PredictionArtifactType.PREDICTIONS,
        storage_type=PredictionStorageType.S3,
        connector_id=_CONNECTOR_ID,
        file_path="predictions.csv",
    )
    response = BatchPredictionJobResponse(
        id=BatchPredictionJobID("batch-prediction-job"),
        project_id=_PROJECT_ID,
        config=BatchPredictionJobConfig(
            predictive_query_id=PredictiveQueryID("pquery"),
            training_job_id=TrainingJobID("training-job"),
            task_type=TaskType.REGRESSION,
            outputs=[output],
            graph_id="graph",
        ),
        timestamps=ResourceTimestamps(
            created_at=now_utc(),
            updated_at=now_utc(),
        ),
        predictive_query_link=PredictiveQueryLink(
            id=PredictiveQueryID("pquery"),
            name=PredictiveQueryID("pquery"),
        ),
        execution=BatchPredictionJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.not_started(),
        ),
        graph_link=GraphLink(id="graph", name="graph"),
    )

    ta = TypeAdapter(BatchPredictionJobResponse)
    # Round-trip through pydantic validation; invalid input would raise
    # ``ValidationError`` (there is no (validated, errors) tuple in v2).
    validated = ta.validate_python(ta.dump_python(response))

    assert isinstance(validated.config.outputs[0], S3PredictionOutput)
