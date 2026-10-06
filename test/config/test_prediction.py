# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import timedelta
from pathlib import Path

from kumo.config.prediction import PartitionedPredictionProgressHelper


def test_partitioned_prediction_progress_helper(tmp_path: Path):
    path = tmp_path / 'test_progress.json'
    helper = PartitionedPredictionProgressHelper(str(path), 10)
    assert helper._progress.completed_partitions == 0
    assert helper._progress.estimated_prediction_time is None

    time.sleep(1)
    helper.update_progress(1)
    assert helper._progress.completed_partitions == 1
    time.sleep(1)
    assert helper._progress.estimated_prediction_time is not None
    assert helper._progress.estimated_prediction_time >= timedelta(seconds=8.9)
    helper.update_progress(2)
    assert helper._progress.completed_partitions == 2
    time.sleep(1)
    assert helper._progress.estimated_prediction_time is not None
    assert helper._progress.estimated_prediction_time >= timedelta(seconds=7.9)
    helper.update_progress(3)
    assert helper._progress.completed_partitions == 3
    assert helper._progress.estimated_prediction_time is not None
    assert helper._progress.estimated_prediction_time >= timedelta(seconds=6.9)
