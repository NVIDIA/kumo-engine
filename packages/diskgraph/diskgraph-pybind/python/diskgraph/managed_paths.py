# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kumo-managed paths for an exact DiskGraph build under one local root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _component(value: str, name: str) -> str:
    if (not value.strip() or value in {".", ".."}
            or "/" in value or "\\" in value or ":" in value):
        raise ValueError(f"{name} must be a single nonempty path component")
    return value


@dataclass(frozen=True)
class ManagedIndexPaths:
    """Resolve the base, task, and cache paths from one configured root.

    The caller obtains ``root`` from Kumo settings and ``task_id`` from the
    FM loader. This helper never chooses an independent task/output root.
    """

    root: Path
    index_build_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        _component(self.index_build_id, "index_build_id")

    @property
    def index_dir(self) -> Path:
        return self.root / "indexes" / self.index_build_id

    def task_dir(self, task_id: str) -> Path:
        return self.root / "tasks" / self.index_build_id / _component(task_id, "task_id")

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache" / self.index_build_id

    def verify_index(self) -> None:
        """Reject a mislabeled or legacy index before attaching new tasks."""
        metadata = json.loads((self.index_dir / "index_metadata.json").read_text())
        if metadata.get("index_build_id") != self.index_build_id:
            raise ValueError(f"index build ID mismatch at {self.index_dir}")

    def verify_task(self, task_id: str) -> None:
        """Check the attachment's exact parent before opening it."""
        self.verify_index()
        metadata = json.loads((self.task_dir(task_id) / "index_metadata.json").read_text())
        if metadata.get("parent_index_build_id") != self.index_build_id:
            raise ValueError(f"task parent index build ID mismatch: {task_id}")
