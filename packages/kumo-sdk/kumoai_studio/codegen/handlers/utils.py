# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import Type


def _get_canonical_import_path(cls: Type) -> str:
    """Dynamically finds the shortest, most canonical import path.

    For example, given the S3Connector class, it would prefer
    'from kumoai_studio import S3Connector' over
    'from kumoai_studio.connector import S3Connector' over
    'from kumoai_studio.connector.s3_connector import S3Connector'.
    """
    base_module_path = cls.__module__
    class_name = cls.__name__

    parts = base_module_path.split(".")

    # The longest path is the one where the class is defined
    canonical_path = base_module_path

    # Walk up the module hierarchy to find a shorter path.
    # e.g., from 'kumoai_studio.connector.s3_connector' to 'kumoai_studio.connector'
    for i in range(len(parts) - 1, 0, -1):
        parent_module_path = ".".join(parts[:i])
        try:
            parent_module = importlib.import_module(parent_module_path)
            if hasattr(parent_module, class_name):
                if getattr(parent_module, class_name) is cls:
                    canonical_path = parent_module_path
                else:
                    # A different object has the same name
                    break
            else:
                # The class isn't in this parent module
                break
        except ImportError:
            # If we can't import a parent, we can't go higher.
            break

    return canonical_path
