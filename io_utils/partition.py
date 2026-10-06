# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for handling partitioned files across different storage
systems."""

import os
from dataclasses import dataclass
from typing import Union

from io_utils import fsutil
from io_utils.fsutil.uri import S3URI, UCURI, LocalURI, ParseURI


@dataclass
class PredTablePartitionPathInfo:
    """Represents the mapping of partitions to their file information.

    This class is used to store and reconstruct file paths for different
    storage systems (S3, UC Volume) that contain partitioned data. It
    maintains the necessary components to reconstruct a file paths template
    for each partition.

    Attributes:
        uri_type: Type of URI (S3URI or UCURI) indicating the storage
            system
        path_prefix: File path up to 'part-' used to construct the path to be
            matched for each partition by adding the partition number and *.
        num_partitions: Total number of partitions in the dataset


    Example:
        For S3 paths (Snowflake is similar):
            path_prefix =  "s3://bucket/path/part-"

        For UC Volume paths:
            path_prefix =  "dbfs:/Volumes/catalog/schema/volume/part-"

    """
    uri_type: type[Union[S3URI, UCURI]]
    path_prefix: str
    num_partitions: int

    def get_prefix_path(self, partition_num: int) -> str:
        """Returns the prefix path for a given partition number, including
        the wildcard character.
        """
        if partition_num >= self.num_partitions or partition_num < 0:
            raise ValueError(f"Partition number out of range: {partition_num}")
        return f"{self.path_prefix}{str(partition_num).zfill(5)}*"


def _process_s3_files(parquet_files: list[str],
                      num_parts: int) -> PredTablePartitionPathInfo:
    """Process S3 parquet files and extract partition information.

    Example filename:
        part-00001-f264aa83-a97b-4e51-8463-d715e38f583b-c000.snappy.parquet

    Args:
        parquet_files: List of parquet file paths in S3

    Returns:
        PredTablePartitionPathInfo: Object containing path prefix and
            partition information for file path reconstruction.

    Raises:
        ValueError: If the file pattern doesn't match the expected EMR Spark
            parquet file pattern or if the file list is empty
    """
    if not parquet_files:
        return PredTablePartitionPathInfo(uri_type=S3URI, path_prefix="",
                                          num_partitions=0)

    first_file = parquet_files[0]
    filename = os.path.basename(first_file)
    file_components = filename.split('-')

    if file_components[0] != 'part' or not file_components:
        raise ValueError(
            f"Unexpected file pattern: {filename}\n"
            "Expected EMR Spark parquet file pattern similar to: "
            "part-<padded_partition_number>-<uuid>-c000.snappy.parquet")

    path_prefix = f"{first_file.rsplit('/', 1)[0]}/part-"

    return PredTablePartitionPathInfo(uri_type=S3URI, path_prefix=path_prefix,
                                      num_partitions=num_parts)


def _process_ucv_files(parquet_files: list[str],
                       num_parts: int) -> PredTablePartitionPathInfo:
    """Process UC Volume parquet files and extract partition information.

    Note: Unlike S3, Spark parquet files outputted into the UC Volume
        also have a task ID at the end of the file name.

    Example filename:
        part-00004-tid-8398707246426104439-2264b669-5641-4216-a42a-096603307e83-9744-1.c000.snappy.parquet

    Args:
        parquet_files: List of parquet file paths in UC Volume

    Returns:
        PredTablePartitionPathInfo: Object containing path prefix and
            partition information for file path reconstruction.

    Raises:
        ValueError: If the file pattern doesn't match the expected Databricks
            parquet file pattern or if the file list is empty
    """
    if not parquet_files:
        return PredTablePartitionPathInfo(uri_type=UCURI, path_prefix="",
                                          num_partitions=0)

    first_file = parquet_files[0]
    filename = os.path.basename(first_file)
    parts = filename.split('-')

    if parts[0] != 'part':
        raise ValueError(f"Unexpected file pattern: {filename}\n"
                         "Expected Databricks parquet file similar to: "
                         "part-<padded_partition>-tid-<uuid>-<task_id>-"
                         "1-c000.snappy.parquet")

    # Extract components dynamically
    partition_id = parts[1]
    filename.find(partition_id) + len(partition_id)

    path_prefix = f"{first_file.rsplit('/', 1)[0]}/part-"

    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if parts[0] == 'part':
            partition_id = int(parts[1])

    return PredTablePartitionPathInfo(uri_type=UCURI, path_prefix=path_prefix,
                                      num_partitions=num_parts)


def _count_and_check_partitions(parquet_files: list[str], path: str) -> int:
    """Returns the count of partitions and raises an error if the partitions
    are not consecutively numbered from zero.
    """
    partition_indices = set()
    max_partition = -1
    for file in parquet_files:
        file_name = os.path.basename(file)
        partition_num = int(file_name.split('-')[1].split('.')[0])
        partition_indices.add(partition_num)
        if partition_num > max_partition:
            max_partition = partition_num
    if len(partition_indices) != max_partition + 1:
        raise ValueError(
            f"Partitions are not consecutively numbered at {path}")
    return max_partition + 1


def get_pred_table_partition_info(path: str) -> PredTablePartitionPathInfo:
    """Returns an object that contains information for file path
        matching.

    This function analyzes a directory containing parquet files and returns an
    object that can be used to prefix-match file paths for each partition.
    It handles Spark files for two types of storage systems:
    1. S3 (EMR Spark parquet files)
    2. UC Volume (Databricks parquet files)

    Assumptions: This function assumes that the parquet files in the directory
        passed in have not been tampered with and only contain the Spark or
        Snowflake parquet files. A partition may consist of multiple files,
        but all parts must have the same file name prefix and parquet
        extension, e.g. part-<partition_num>*.parquet. Partition numbers must
        be consecutive starting from zero. The only hard limit on number of
        partitions is 99999 but the expected scale here is up to ~100.

    Args:
        path: Path to directory containing parquet files

    Returns:
        PredTablePartitionPathInfo: Object containing path and partition
            information for file path reconstruction.

    Raises:
        ValueError: If the path is not a directory or if the URI type is not
            supported
    """
    if not fsutil.isdir(path):
        raise ValueError(f"Expected a directory, got {path}")

    parquet_files = fsutil.listdir(path, suffix=".parquet")
    num_parts = _count_and_check_partitions(parquet_files, path)
    uri = ParseURI(path)

    # Local PySpark and EMR Spark generate files with the same pattern
    if isinstance(uri, S3URI) or isinstance(uri, LocalURI):
        return _process_s3_files(parquet_files, num_parts)
    elif isinstance(uri, UCURI):
        return _process_ucv_files(parquet_files, num_parts)
    else:
        raise ValueError(f"Unsupported URI type: {type(uri)}")
