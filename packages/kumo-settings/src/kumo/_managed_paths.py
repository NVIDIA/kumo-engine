# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private Kumo layout for a DiskGraph build under the configured root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _component(value: str, name: str) -> str:
    if (not value.strip() or value in {".", ".."}
            or "/" in value or "\\" in value or ":" in value):
        raise ValueError(f"{name} must be a single nonempty path component")
    return value


@dataclass(frozen=True)
class _ManagedIndexPaths:
    """Derive build paths from an explicit root already resolved by the caller."""

    index_build_id: str
    root: Path

    def __post_init__(self) -> None:
        _component(self.index_build_id, "index_build_id")

    @property
    def managed_root(self) -> Path:
        return self.root

    @property
    def index_dir(self) -> Path:
        return self.managed_root / "indexes" / self.index_build_id

    def task_dir(self, task_id: str) -> Path:
        return self.managed_root / "tasks" / self.index_build_id / _component(task_id, "task_id")

    @property
    def cache_dir(self) -> Path:
        return self.managed_root / "cache" / self.index_build_id
