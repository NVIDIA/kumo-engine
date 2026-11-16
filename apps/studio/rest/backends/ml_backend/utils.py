# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from typing import Optional

from rest.config.file_type import FileType

ONE_BILLION = 1e9


def validated_file_type(file_path: str,
                        file_type: Optional[FileType]) -> FileType:
    if file_type is not None:
        return file_type

    # Try to parse it from the file path.
    try:
        return FileType.parse_extension(file_path)
    except ValueError:
        # Otherwise return parquet
        return FileType.PARQUET


def canonical_file_path(file_path: str) -> str:
    """Returns canonical string that can be used to detect conflicts
    between files/directories that have different names, but refer to the
    same s3 location.
    """
    return file_path.rstrip('/')


def is_partitioned(file_path: str) -> bool:
    return file_path.endswith('/')


def directory_name(file_path: str) -> str:
    if is_partitioned(file_path):
        return file_path
    return osp.dirname(file_path) + '/'
