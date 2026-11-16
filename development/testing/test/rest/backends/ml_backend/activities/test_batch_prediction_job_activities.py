# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.train.prediction_partitions import split_partitions_across_jobs


def test_split_partitions_across_jobs():
    # Test for partition_count < job_count.
    with pytest.raises(ValueError):
        split_partitions_across_jobs(5, 6)

    # Test for partition_count > job_count.
    assert split_partitions_across_jobs(12, 5) == [(0, 3), (3, 6), (6, 8),
                                                   (8, 10), (10, 12)]

    # Test for partition_count == job_count.
    assert split_partitions_across_jobs(5, 5) == [(0, 1), (1, 2), (2, 3),
                                                  (3, 4), (4, 5)]
