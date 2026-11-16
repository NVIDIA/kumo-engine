# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for normalizing persist-directory paths across
local and cloud storage backends.
"""
import os

from io_utils.fsutil import is_local


def normalize_persist_dir(path: str) -> str:
    r"""Normalize a persist directory path to its canonical form.

    Local paths are expanded via ``os.path.expanduser``; S3 / UCV /
    other cloud storage paths are returned as-is (the DG /
    PyArrow runtime consumes ``s3://`` directly).

    Args:
        path (str): The persist directory path.

    Returns:
        str: The normalized path.
    """
    if is_local(path):
        return os.path.expanduser(path)
    return path
