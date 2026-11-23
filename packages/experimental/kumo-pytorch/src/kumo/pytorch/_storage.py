# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage policy for managed DiskGraph indexes and derived state."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import kumo.settings
from kumo._managed_paths import _ManagedIndexPaths


def managed_root() -> Path:
    """Capture the configured storage root for one engine."""
    return kumo.settings.scratch_dir().resolve()


def managed_paths(index_build_id: str | None = None, *, root: Path | None = None):
    """Resolve one physical build under the process-wide Kumo scratch root."""
    try:
        from diskgraph import new_index_build_id
    except ImportError as error:
        raise RuntimeError(
            "Managed indexes require DiskGraph build-identity support; install "
            "DiskGraph from the matching kumo-engine revision"
        ) from error
    return _ManagedIndexPaths(
        new_index_build_id() if index_build_id is None else index_build_id,
        root=managed_root() if root is None else root,
    )


def verify_managed_roots(paths, index: str | Path) -> None:
    """Keep consumer writes in this build's managed task and cache roots."""
    base = Path(index).resolve()
    for destination in (paths.task_dir("probe").parent, paths.cache_dir):
        resolved = destination.resolve()
        if resolved.is_relative_to(base) or base.is_relative_to(resolved):
            raise ValueError("managed task/cache paths must be disjoint from the base index")
        if resolved != destination:
            raise ValueError("managed task/cache paths must not redirect to another location")
    stats = paths.cache_dir / "col_stats"
    if stats.resolve() != stats:
        raise ValueError("managed col_stats path must not redirect to another location")


def verify_task_manifest(
    manifest: Mapping[str, Any], *, paths: _ManagedIndexPaths, index: str | Path, task_id: str
) -> None:
    """Validate Kumo's manifest placement before ingest and native handoff.

    Layout comes from the captured managed paths. DiskGraph owns the separate
    extension-to-parent build identity check.
    """
    def verify_path(value: str, expected: Path, field: str) -> None:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{field} must be an absolute path without traversal")
        if path != expected or path.resolve() != path:
            raise ValueError(f"{field} must match its bound path without redirects: {expected}")

    base = Path(index)
    if not base.is_relative_to(paths.root):
        raise ValueError("manifest graph.index_path must be within the captured Kumo root")
    verify_path(manifest["graph"]["index_path"], base, "manifest graph.index_path")
    task = paths.task_dir(task_id)
    for name, stage in manifest["stages"].items():
        verify_path(stage["index_path"], task, f"manifest stages[{name!r}].index_path")


class TaskDirectory:
    """An exclusively owned loader task directory; never adopt another task's files."""

    def __init__(self, paths, task_id: str | None = None):
        import shutil
        import uuid
        import weakref

        self.task_id = uuid.uuid4().hex if task_id is None else task_id
        path = paths.task_dir(self.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # Refuse a live task or leftover state with the same ID.
        self.name = str(path)
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.name, ignore_errors=True)

    def cleanup(self):
        self._finalizer()


class ColumnCache:
    """Read base caches and publish missing entries to the requested root."""

    def __init__(self, index: str | Path, root: str | Path | None = None):
        self.index = Path(index)
        self.root = self.index if root is None else Path(root)
        self.external = self.root.resolve() != self.index.resolve()
        if not self.external:
            self.root = self.index

    def destination(self, name: str) -> Path:
        destination = self.root / "col_stats" / name
        if self.external and destination.resolve() != destination:
            raise ValueError("external col_stats path must not redirect to another location")
        return destination

    def candidates(self, name: str, *, base_name: str | None = None) -> tuple[Path, ...]:
        base = self.index / "col_stats" / (name if base_name is None else base_name)
        return tuple(dict.fromkeys((base, self.destination(name))))

    def resolve(self, name: str) -> Path:
        return next(
            (path for path in self.candidates(name) if path.is_file()), self.destination(name)
        )

    def reference(self, path: Path) -> str:
        return str(path if self.external else path.relative_to(self.index / "col_stats"))
