# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private Kumo layout for local snapshots and DiskGraph builds."""

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


@dataclass(frozen=True)
class _SnapshotPaths:
    """Local snapshot names, including compatibility with a custom legacy root."""

    data_root: Path
    _reference_suffix = ".index.json"

    @classmethod
    def for_root(cls, root: Path) -> _SnapshotPaths:
        return cls(root / "snapshots")

    def graph_dir(self, graph_key: str) -> Path:
        return self.data_root / _component(graph_key, "graph_key")

    def version_dir(self, graph_key: str, version: str) -> Path:
        return self.graph_dir(graph_key) / _component(version, "snapshot version")

    def reference(self, graph_key: str, version: str) -> Path:
        path = self.version_dir(graph_key, version)
        return path.with_name(path.name + self._reference_suffix)

    def publication_reference(self, graph_key: str, version: str) -> Path:
        if version.endswith(self._reference_suffix):
            raise ValueError("managed snapshot versions must not end with "
                             + self._reference_suffix)
        return self.reference(graph_key, version)

    @classmethod
    def reference_version(cls, name: str) -> str | None:
        """Return the logical version encoded by a reference filename."""
        if not name.endswith(cls._reference_suffix):
            return None
        version = name.removesuffix(cls._reference_suffix)
        try:
            return _component(version, "snapshot version")
        except ValueError:
            return None
