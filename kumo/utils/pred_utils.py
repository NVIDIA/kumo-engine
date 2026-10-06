#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from typing import List, Tuple

from kumo_core.job_config import PredictionConfig

from io_utils import logging, parquet_util
from io_utils.partition import get_pred_table_partition_info
from kumo.connector import ParquetConnector
from kumo.pquery.predictive_query import train_table_source_name

logger = logging.getLogger(__name__)


def get_partition_offsets_and_lengths(
        pred_table_path: str) -> Tuple[List[int], List[int]]:
    """
    Get offsets and lengths for each partition based on row counts.
    Returns two lists: offsets and lengths.
    """
    pred_partition_info = get_pred_table_partition_info(pred_table_path)
    logger.info("Get row counts for each partition in the prediction table")
    num_partitions = pred_partition_info.num_partitions

    row_counts = []
    for i in range(num_partitions):
        # Note - pred_partition_info checks existince of parts for each prefix
        partition_path = pred_partition_info.get_prefix_path(i)
        # Read metadata for the partition
        partition_metadata = parquet_util.get_parquet_metadata(partition_path)
        row_counts.append(partition_metadata.num_rows)

    logger.info(
        f"Total rows: {sum(row_counts)}, Row counts per partition: {row_counts}"
    )
    offsets = []
    lengths = []
    current_offset = 0

    for row_count in row_counts:
        offsets.append(current_offset)
        lengths.append(row_count)
        current_offset += row_count

    return offsets, lengths


def get_pred_table_connector(
    pred_config: PredictionConfig, ) -> ParquetConnector:
    """
    Returns the ParquetConnector for the pred table.
    """
    pred_table_path = pred_config.pred_table_path
    return ParquetConnector(name=train_table_source_name(pred_table_path),
                            root_dir=osp.dirname(pred_table_path))
