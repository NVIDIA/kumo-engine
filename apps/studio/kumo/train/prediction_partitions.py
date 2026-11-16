# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np
from kumo_core.job_config import PredictionConfig


def split_partitions_across_jobs(
    num_partitions: int,
    num_jobs: int,
) -> List[Tuple[int, int]]:
    """Return balanced contiguous ``[start, stop)`` partition ranges."""
    if num_partitions < num_jobs:
        raise ValueError(f"num_partitions: {num_partitions} must be >= "
                         f"num_jobs: {num_jobs}")
    partition_ranges = np.array_split(np.arange(num_partitions), num_jobs)
    # The stop partition is exclusive. Convert numpy.int64 values to int.
    return [(partition_range[0].item(), partition_range[-1].item() + 1)
            for partition_range in partition_ranges]


def prediction_output_partition_range(
    config: PredictionConfig, ) -> Tuple[int, int]:
    """Return the logical output range written by one prediction runtime.

    TT-NG uses ``start_partition_id`` as the runtime worker identity while
    output filenames use a separate split of
    ``effective_output_num_partitions``. Legacy prediction configs retain
    their direct input/output partition range.
    """
    if not config.diskgraph_task_dataset_bundle_dir:
        return config.start_partition_id, config.stop_partition_id
    if config.stop_partition_id != config.start_partition_id + 1:
        raise ValueError(
            "TT-NG prediction config must select one runtime shard")
    ranges = split_partitions_across_jobs(
        config.effective_output_num_partitions,
        config.num_partitions,
    )
    return ranges[config.start_partition_id]


def prediction_output_partition_count(config: PredictionConfig) -> int:
    """Return the complete logical output-partition union for ``config``."""
    return (config.effective_output_num_partitions if
            config.diskgraph_task_dataset_bundle_dir else config.num_partitions)
