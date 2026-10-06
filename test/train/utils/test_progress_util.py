# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta

from kumoapi.jobs import PredictionProgress

from kumo.config.prediction import PartitionedPredictionProgress
from kumo.train.utils.progress_util import get_partition_progress


def test_get_partition_progress():
    """Test different cases for the `get_partition_progress` function."""
    # Case 1: partition progress is None
    partition_progress = None
    prediction_progress = PredictionProgress(
        start_time=datetime.utcnow(), total_iterations=10,
        completed_iterations=5, elapsed_prediction_time=timedelta(seconds=10),
        estimated_prediction_time=timedelta(seconds=20))
    expected = prediction_progress
    actual = get_partition_progress(partition_progress, prediction_progress)
    assert actual == expected

    # Case 2: prediction progress is not None and prediction progress is not
    # completed
    partition_progress = PartitionedPredictionProgress(
        start_time=datetime.utcnow(), total_partitions=2,
        completed_partitions=1,
        estimated_prediction_time=timedelta(seconds=30))
    prediction_progress = PredictionProgress(
        start_time=datetime.utcnow(), total_iterations=10,
        completed_iterations=5, elapsed_prediction_time=timedelta(seconds=10),
        estimated_prediction_time=None)
    expected = PredictionProgress(
        start_time=partition_progress.start_time, total_iterations=20,
        completed_iterations=15, elapsed_prediction_time=timedelta(seconds=10),
        estimated_prediction_time=timedelta(seconds=20))
    actual = get_partition_progress(partition_progress, prediction_progress)
    assert actual == expected

    # Case 3: partition progress is not None
    partition_progress = PartitionedPredictionProgress(
        start_time=datetime.utcnow(), total_partitions=2,
        completed_partitions=1,
        estimated_prediction_time=timedelta(seconds=30))
    prediction_progress = PredictionProgress(
        start_time=datetime.utcnow(), total_iterations=10,
        completed_iterations=5, elapsed_prediction_time=timedelta(seconds=10),
        estimated_prediction_time=timedelta(seconds=20))
    expected = PredictionProgress(
        start_time=partition_progress.start_time, total_iterations=20,
        completed_iterations=15, elapsed_prediction_time=timedelta(seconds=10),
        estimated_prediction_time=timedelta(seconds=20))
    actual = get_partition_progress(partition_progress, prediction_progress)
    assert actual == expected
