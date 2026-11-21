# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private Kumo layout for a DiskGraph build under the configured root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kumo.settings


def _component(value: str, name: str) -> str:
    if (not value.strip() or value in {".", ".."}
            or "/" in value or "\\" in value or ":" in value):
        raise ValueError(f"{name} must be a single nonempty path component")
    return value


@dataclass(frozen=True)
class _ManagedIndexPaths:
    index_build_id: str

    def __post_init__(self) -> None:
        _component(self.index_build_id, "index_build_id")

    @property
    def index_dir(self) -> Path:
        return kumo.settings.scratch_dir() / "indexes" / self.index_build_id

    def task_dir(self, task_id: str) -> Path:
        return kumo.settings.scratch_dir() / "tasks" / self.index_build_id / _component(task_id, "task_id")

    @property
    def cache_dir(self) -> Path:
        return kumo.settings.scratch_dir() / "cache" / self.index_build_id
