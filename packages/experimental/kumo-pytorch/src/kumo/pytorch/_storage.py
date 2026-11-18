# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage policy for index-local and external consumer state."""

from __future__ import annotations

import hashlib
from pathlib import Path


def normalize_work_dir(
    index: str | Path, work_dir: str | Path | None, *, external_only: bool = False
) -> Path | None:
    """Use the legacy layout for an omitted or explicitly equal index path."""
    base = Path(index).resolve()
    work = None if work_dir is None else Path(work_dir).resolve()
    if not external_only and (work is None or work == base):
        return None
    if work is None or work.is_relative_to(base):
        raise ValueError("work_dir must be outside the base index")
    return work


class ColumnCache:
    """Read immutable base caches and publish missing entries to the writable root."""

    def __init__(self, index: str | Path, root: str | Path | None = None):
        self.index = Path(index)
        self.root = self.index if root is None else Path(root)
        self.external = self.root.resolve() != self.index.resolve()

    @classmethod
    def for_work_dir(cls, index: str | Path, work_dir: Path | None, identity: str):
        root = (
            None
            if work_dir is None
            else work_dir / "cache" / hashlib.sha256(identity.encode()).hexdigest()
        )
        return cls(index, root)

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
