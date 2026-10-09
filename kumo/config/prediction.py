# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta

from kumo_core.utils.time import now_utc

from io_utils import kumo_dataclass
from io_utils.json import save_json


@kumo_dataclass
class PartitionedPredictionProgress:
    """Provide partition level prediction progress.
    """
    start_time: datetime = now_utc()
    total_partitions: int = 0
    completed_partitions: int = 0
    elapsed_prediction_time: timedelta = timedelta(seconds=0)
    # Provides an estimated amount of time for the prediction to complete.
    # This is generally an upper bound and the estimate gets more accurate
    # as time progresses.
    estimated_prediction_time: timedelta | None = None


class PartitionedPredictionProgressHelper():
    def __init__(self, path: str, total_partitions: int):
        self._path = path
        self._progress = PartitionedPredictionProgress(
            start_time=now_utc(),
            total_partitions=total_partitions,
            completed_partitions=0,
        )
        self._save_progress()
        self._last_update_time = now_utc()

    def _save_progress(self):
        save_json(self._progress, self._path)

    def update_progress(self, completed_partitions: int):
        self._progress.completed_partitions = completed_partitions

        now = now_utc()
        self._progress.elapsed_prediction_time = (now -
                                                  self._progress.start_time)
        average_time_per_partition = (self._progress.elapsed_prediction_time /
                                      self._progress.completed_partitions)
        remaining_partitions = (self._progress.total_partitions -
                                self._progress.completed_partitions)
        self._progress.estimated_prediction_time = (average_time_per_partition *
                                                    remaining_partitions)
        self._last_update_time = now

        self._save_progress()
