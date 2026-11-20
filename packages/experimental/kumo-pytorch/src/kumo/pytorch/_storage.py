# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage policy for index-local and external consumer state."""

from __future__ import annotations

from pathlib import Path


def managed_paths(index_build_id: str | None = None):
    """Resolve one physical build under the process-wide Kumo scratch root."""
    try:
        from diskgraph import new_index_build_id
        from diskgraph.managed_paths import ManagedIndexPaths
    except ImportError as error:
        raise RuntimeError(
            "Managed indexes require DiskGraph build-identity support; install "
            "DiskGraph from the matching kumo-engine revision"
        ) from error
    import kumo.settings

    return ManagedIndexPaths(
        kumo.settings.scratch_dir().resolve(),
        new_index_build_id() if index_build_id is None else index_build_id,
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


class TaskDirectory:
    """An exclusively owned loader task directory; never adopt another task's files."""

    def __init__(self, paths, task_id: str | None = None):
        import uuid

        self.task_id = uuid.uuid4().hex if task_id is None else task_id
        path = paths.task_dir(self.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # Refuse a live task or leftover state with the same ID.
        self.name = str(path)

    def cleanup(self):
        import shutil

        shutil.rmtree(self.name, ignore_errors=True)


class ColumnCache:
    """Read immutable base caches and publish missing entries to the writable root."""

    def __init__(self, index: str | Path, root: str | Path | None = None):
        self.index = Path(index)
        self.root = self.index if root is None else Path(root)
        self.external = self.root.resolve() != self.index.resolve()
        if not self.external:
            self.root = self.index

    def destination(self, name: str) -> Path:
        return self.root / "col_stats" / name

    def candidates(self, name: str, *, base_name: str | None = None) -> tuple[Path, ...]:
        base = self.index / "col_stats" / (name if base_name is None else base_name)
        return tuple(dict.fromkeys((base, self.destination(name))))

    def resolve(self, name: str) -> Path:
        return next(
            (path for path in self.candidates(name) if path.is_file()), self.destination(name)
        )

    def reference(self, path: Path) -> str:
        return str(path if self.external else path.relative_to(self.index / "col_stats"))
