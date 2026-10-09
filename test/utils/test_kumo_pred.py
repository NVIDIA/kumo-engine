# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import tempfile

import pandas as pd

from kumo.utils.pred_utils import get_partition_offsets_and_lengths


def test_get_partition_offsets_and_lengths():
    with tempfile.TemporaryDirectory() as tmpdir:
        partitioned_pred_table_path = osp.join(str(tmpdir),
                                               'partitioned_pred_table')
        os.makedirs(partitioned_pred_table_path)
        partition_lengths = [111, 111, 111, 111, 111, 111, 112, 111, 111]
        for partition, length in enumerate(partition_lengths):
            path = osp.join(
                partitioned_pred_table_path,
                f'part-{partition:05d}-test-c000.snappy.parquet',
            )
            pd.DataFrame({'id': range(length)}).to_parquet(path)

        offsets, lengths = get_partition_offsets_and_lengths(
            partitioned_pred_table_path)
        assert offsets == [0, 111, 222, 333, 444, 555, 666, 778, 889]
        assert lengths == partition_lengths
