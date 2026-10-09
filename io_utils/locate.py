# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import Any


def locate(path: str) -> Any:
    r"""Locates an object by dotted path."""
    parts = path.split('.')

    if len(parts) <= 1:
        raise ValueError(f"Path is missing a dot (got '{path}').")

    for part in parts:
        if len(part) == 0:
            raise ValueError(f"Relative imports not supported (got '{path}').")

    module, name = '.'.join(parts[:-1]), parts[-1]
    module = import_module(module)
    return getattr(module, name)
