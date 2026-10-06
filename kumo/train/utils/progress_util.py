# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta

from kumoapi.jobs import PredictionProgress

from io_utils import fsutil, logging
from io_utils.json import load_json
from kumo.config.prediction import PartitionedPredictionProgress

log = logging.getLogger(__name__)

FILE_PREFIX = "prediction_progress_"
PARTITION_FILE_PREFIX = "partitioned_progress_"


def get_partition_progress(
    partition_progress: PartitionedPredictionProgress | None,
    prediction_progress: PredictionProgress,
) -> PredictionProgress:
    if partition_progress is None:
        # Partition progress initialized before prediction progress. When it is
        # None, it is single partition, return prediction progress.
        return prediction_progress

    total_iterations = (prediction_progress.total_iterations *
                        partition_progress.total_partitions)
    completed_iterations = (prediction_progress.completed_iterations +
                            prediction_progress.total_iterations *
                            partition_progress.completed_partitions)
    if partition_progress.estimated_prediction_time is None:
        # Running the first partition, estimate based on total iterations.
        elapsed_prediction_time = prediction_progress.elapsed_prediction_time
        per_iteration_speed = elapsed_prediction_time / \
            (prediction_progress.completed_iterations + 1e-6)
        return PredictionProgress(
            start_time=partition_progress.start_time,
            total_iterations=total_iterations,
            completed_iterations=completed_iterations,
            elapsed_prediction_time=elapsed_prediction_time,
            estimated_prediction_time=per_iteration_speed *
            (total_iterations - completed_iterations))

    # Running several partitions, estimate based on combination of partition
    # progress and prediction progress.
    if (prediction_progress.completed_iterations ==
            prediction_progress.total_iterations):
        # Prediction progress is completed, use partition progress only.
        elapsed_prediction_time = timedelta(seconds=0)
    else:
        elapsed_prediction_time = prediction_progress.elapsed_prediction_time
    return PredictionProgress(
        start_time=partition_progress.start_time,
        total_iterations=total_iterations,
        completed_iterations=completed_iterations,
        elapsed_prediction_time=elapsed_prediction_time,
        estimated_prediction_time=(
            partition_progress.estimated_prediction_time -
            elapsed_prediction_time))


def get_prediction_progress(log_dir: str) -> PredictionProgress | None:
    if not log_dir.endswith(os.path.sep):
        log_dir += os.path.sep

    all_paths = fsutil.listdir(log_dir)

    def _strip_ext(name: str) -> str:
        # Keep behavior robust whether ".json" is present or not
        return name[:-5] if name.endswith(".json") else name

    # Collect per-partition files keyed by partition_id (number as string)
    partitioned_progress_dict: dict[str, PartitionedPredictionProgress] = {}
    prediction_progress_dict: dict[str, PredictionProgress] = {}

    for path in all_paths:
        fname = os.path.basename(path)
        if fname.startswith(
                PARTITION_FILE_PREFIX):  # e.g., "partitioned_progress_"
            partition_id = _strip_ext(fname[len(PARTITION_FILE_PREFIX):])
            partitioned_progress_dict[partition_id] = load_json(
                path, PartitionedPredictionProgress)
        elif fname.startswith(FILE_PREFIX):  # e.g., "prediction_progress_"
            partition_id = _strip_ext(fname[len(FILE_PREFIX):])
            prediction_progress_dict[partition_id] = load_json(
                path, PredictionProgress)

    # --- Primary path: derive overall progress from partitioned progress ---
    if partitioned_progress_dict and prediction_progress_dict:
        # Combine per-partition "main" (partitioned) with matching "within-partition" progress
        worst_progress: PredictionProgress | None = None
        worst_eta = timedelta(seconds=0)
        any_with_eta = False

        # Iterate only over partitions that have partitioned progress
        for partition_id, part_main in partitioned_progress_dict.items():
            part_inner = prediction_progress_dict.get(partition_id, None)

            if not part_inner:
                continue

            combined = get_partition_progress(part_main, part_inner)
            # combined is expected to be a PredictionProgress-like object

            eta = combined.estimated_prediction_time
            if eta is not None:
                any_with_eta = True
                if eta >= worst_eta:
                    worst_eta = eta
                    worst_progress = combined

        # If at least one partition had an ETA, return the slowest (conservative overall)
        if any_with_eta and worst_progress is not None:
            return worst_progress

        # If none had ETA yet, we can't reliably report progress
        return None

    # --- Fallback: single global progress file (legacy path) ---
    if len(prediction_progress_dict) > 0:
        # Only one prediction progress is available, if not, log a warning.
        if len(prediction_progress_dict) > 1:
            log.warning("More then one prediction progress is ready "
                        f"{prediction_progress_dict}")
        for value in prediction_progress_dict.values():
            return value

    # Nothing useful found
    return None
