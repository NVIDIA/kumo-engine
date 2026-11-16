# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core-owned training workflows.

Contracts are imported from their owning modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_config import TaskConfig

__all__ = ["TaskConfig"]


def __getattr__(name: str):
    if name == "TaskConfig":
        from .task_config import TaskConfig

        return TaskConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
