# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single native training-table publication for persistence and replication."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import orjson
from kumo_core.diskgraph.sampling_artifacts import (
    ArtifactBundle,
    ArtifactFile,
    download_artifact,
    relative_artifact_path,
    relocate_artifacts,
)
from kumo_core.io.json import from_json_dict, to_json_dict
from kumo_core.model_plan import TrainingTableGenerationPlan

if TYPE_CHECKING:
    from kumo.engine._mlops import MLOps
    from kumo.engine.connector import Connector
    from kumo.engine.graph import GraphSnapshot
    from kumo.engine.pquery import PredictiveQuery, TrainingTable

_MARKER = "kumo_training_table.json"
_GENERATION = re.compile(r"[0-9a-f]{32}")


def store_native(
    table: TrainingTable,
    settings: MLOps,
    on_file_uploaded: Callable[[str, ArtifactFile], None] | None = None,
    on_files_uploaded: Callable[[], None] | None = None,
) -> None:
    """Persist native staging once and notify replicas from the same uploads.

    Args:
        table: Immutable generated rows and native ingestion sidecars.
        settings: MLOps settings supplying the canonical training-table root.
        on_file_uploaded: Notification after each successful file close.
        on_files_uploaded: Optional replica barrier before committing the marker.
    Raises:
        ValueError: If staging paths are invalid or staging is empty.
        OSError: If an upload or marker write fails. Callback failures propagate.
    """
    from kumo.engine._snapshot_io import _location, _write, upload_files
    from pyarrow.fs import FileType

    source = table.staging_dir.resolve(strict=True)
    path = settings.root.location(f"training_tables/{table.id}")
    filesystem, root = _location(settings.root.connector, path)
    if (
        on_file_uploaded is None
        and filesystem.get_file_info(f"{root}/_SUCCESS").type is FileType.File
    ):
        return
    train_table_path = table.train_table_path.resolve(strict=True).relative_to(source)
    manifest_path = table.manifest_path.resolve(strict=True).relative_to(source)
    generation = uuid4().hex
    prefix = f"{root}/native/{generation}"
    inventory = _inventory(source, (source,))

    def notify(directory: str, entry: ArtifactFile) -> None:
        if on_file_uploaded is not None:
            on_file_uploaded(directory, entry)

    entries = upload_files(
        settings.root.connector,
        source,
        f"{path}/native/{generation}/files",
        inventory,
        notify,
    )
    if on_files_uploaded is not None:
        on_files_uploaded()
    marker = orjson.dumps(
        {
            "format": 1,
            "generation": generation,
            "source_root": str(table.staging_dir),
            "id": table.id,
            "query": table.query,
            "train_table_path": train_table_path.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "required_stages": table.required_stages,
            "generation_plan": to_json_dict(table.generation_plan),
            "files": [asdict(entry) for entry in entries],
        }
    )
    if filesystem.type_name == "local":
        staged = f"{prefix}/{_MARKER}"
        _write(filesystem, staged, marker)
        filesystem.move(staged, f"{root}/{_MARKER}")
    else:
        _write(filesystem, f"{root}/{_MARKER}", marker)
    for name, payload in table._sidecars().items():
        _write(filesystem, f"{root}/{name}", payload)
    _write(filesystem, f"{root}/_SUCCESS", b"")


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
        settings.root.connector, root, f"{path}/files", inventory, on_file_uploaded
    )
    on_files_uploaded()
    bundle = ArtifactBundle(str(root), entries)
    _write(filesystem, f"{prefix}/manifest.json", orjson.dumps(asdict(bundle)))


def load_native(
    connector: Connector,
    path: str,
    pquery: PredictiveQuery,
    snapshot: GraphSnapshot,
) -> TrainingTable | None:
    """Restore native MLOps staging, or let callers load a legacy portable export.

    Args:
        connector: Connector owning the file destination.
        path: Canonical training-table directory on the connector.
        pquery: Original query whose generated rows are being restored.
        snapshot: Compatible graph snapshot selected for future ingestion.
    Returns:
        Restored table with verified local files, or None without a native marker.
    Raises:
        ValueError: If the marker, query, inventory, or downloaded bytes are invalid.
        OSError: If a persisted file cannot be read or installed locally.
    """
    from kumo.engine._scratch import new_run_id, train_table_scratch_dir
    from kumo.engine.connector import file_store
    from kumo.engine.pquery.training_table import TrainingTable

    location = file_store(connector, path)
    if location is None:
        return None
    filesystem, root = location
    try:
        with filesystem.open_input_stream(f"{root.rstrip('/')}/{_MARKER}") as reader:
            marker = orjson.loads(reader.read())
    except (FileNotFoundError, NotADirectoryError):
        return None
    if (
        not isinstance(marker, dict)
        or marker.get("format") != 1
        or not isinstance(marker.get("generation"), str)
        or not _GENERATION.fullmatch(marker["generation"])
    ):
        raise ValueError("Unsupported native training-table publication")
    pquery._require_matching_snapshot(snapshot)
    if marker["query"] != pquery.query:
        raise ValueError("Native training-table publication belongs to another query")
    entries = tuple(ArtifactFile(**entry) for entry in marker["files"])
    if not entries or len({entry.name for entry in entries}) != len(entries):
        raise ValueError("Empty or duplicate native training-table inventory")
    for entry in entries:
        relative_artifact_path(entry.name)
        if entry.size < 0 or re.fullmatch(r"[0-9a-f]{64}", entry.sha256) is None:
            raise ValueError("Invalid native training-table file identity")
    train_table = relative_artifact_path(marker["train_table_path"])
    manifest = relative_artifact_path(marker["manifest_path"])
    destination = train_table_scratch_dir(new_run_id())
    destination.mkdir(parents=True, exist_ok=False)
    prefix = f"{root.rstrip('/')}/native/{marker['generation']}"
    from kumo_core.sources.object_store import copies
    from kumo_core.sources.parallel import run_in_parallel

    with copies(entries) as workers:
        run_in_parallel(
            [
                partial(
                    download_artifact, filesystem, f"{prefix}/files", entry, destination
                )
                for entry in entries
            ],
            workers=workers,
            stop=threading.Event(),
        )
    relocate_artifacts(ArtifactBundle(marker["source_root"], entries), destination)
    if (
        not (destination / train_table).exists()
        or not (destination / manifest).is_file()
    ):
        raise ValueError(
            "Native training-table inventory is missing its table or manifest"
        )
    return TrainingTable(
        query=marker["query"],
        train_table_path=destination / train_table,
        manifest_path=destination / manifest,
        staging_dir=destination,
        required_stages=tuple(marker["required_stages"]),
        generation_plan=from_json_dict(
            marker["generation_plan"], TrainingTableGenerationPlan
        ),
        id=marker["id"],
    )
