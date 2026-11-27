# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publication of fit-specific ingested extensions and task metadata."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from kumo_core.diskgraph.sampling_artifacts import ArtifactBundle, ArtifactFile

if TYPE_CHECKING:
    from kumo.engine._mlops import MLOps


def _inventory(root: Path, directories: Sequence[Path]) -> dict[str, int]:
    root = root.resolve(strict=True)
    inventory = {
        item.relative_to(root).as_posix(): item.stat().st_size
        for directory in directories
        for item in directory.resolve().rglob("*")
        if item.is_file()
    }
    if not inventory:
        raise ValueError("Cannot publish missing or empty artifacts")
    return inventory


def store_prepared(
    root: Path,
    directories: Sequence[Path],
    settings: MLOps,
    on_file_uploaded: Callable[[str, ArtifactFile], None],
    on_files_uploaded: Callable[[], None],
) -> None:
    """Publish ingested table extensions and task metadata through MLOps.

    Args:
        root: Common local root of the prepared graph and task artifacts.
        directories: Completed extension and fit metadata directories.
        settings: Effective MLOps configuration for this ingestion call.
        on_file_uploaded: Notification after each upload closes successfully.
        on_files_uploaded: Replica readiness barrier before publication commits.
    Raises:
        ValueError: If directories are empty or contain unsafe file paths.
        OSError: If upload or marker publication fails. Callback failures propagate.
    """
    from kumo.engine._snapshot_io import _location, _write, upload_files

    inventory = _inventory(root, directories)
    path = settings.root.location(f"training_tables/ingested/{uuid4().hex}")
    filesystem, prefix = _location(settings.root.connector, path)
    entries = upload_files(
        settings.root.connector,
        root,
        f"{path}/files",
        inventory,
        on_file_uploaded,
    )
    on_files_uploaded()
    bundle = ArtifactBundle(str(root), entries)
    _write(
        filesystem,
        f"{prefix}/manifest.json",
        json.dumps(asdict(bundle)).encode("utf-8"),
    )
