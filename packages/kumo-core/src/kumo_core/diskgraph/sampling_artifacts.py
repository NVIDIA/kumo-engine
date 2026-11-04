# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified artifact downloads and worker-local metadata relocation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

import orjson
from pyarrow.fs import FileSystem

_CHUNK_BYTES = 8 << 20


@dataclass(frozen=True)
class ArtifactFile:
    """Identity of one fully uploaded file.

    Args:
        name: Relative POSIX name below the local and remote index roots.
        size: Uploaded byte count.
        sha256: Digest computed while streaming the uploaded bytes.
    """

    name: str
    size: int
    sha256: str


def relative_artifact_path(name: str) -> Path:
    """Validate a file name before joining it to a graph or artifact root.

    Args:
        name: Nonempty, normalized relative POSIX file name.
    Returns:
        A relative local path with the same components.
    Raises:
        ValueError: If the name is absolute, malformed or escapes its root.
    """
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or ".." in path.parts
        or "\\" in name
    ):
        raise ValueError("Artifact names must stay below the graph root")
    return Path(name)


@dataclass(frozen=True)
class ArtifactBundle:
    """Committed file inventory and original root for metadata relocation.

    Args:
        source_root: Producer-local artifact root referenced by JSON metadata.
        files: Complete inventory in transfer scheduling order.
    """

    source_root: str
    files: tuple[ArtifactFile, ...]


def download_artifact(
    filesystem: FileSystem, directory: str, entry: ArtifactFile, root: Path
) -> None:
    """Download and verify an object before atomically installing it locally.

    Args:
        filesystem: Authenticated artifact-root filesystem.
        directory: Storage directory containing the uploaded relative file names.
        entry: Expected relative name, size and SHA-256.
        root: Worker-local graph directory, never a shared network mount.
    Raises:
        ValueError: If the destination escapes root or contents do not match.
        OSError: If transfer or atomic installation fails. Partial files remain
            under unique .partial names for diagnosis and are never served.
    """
    target = root / relative_artifact_path(entry.name)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact destination escapes the graph root")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{uuid4().hex}.partial")
    digest = hashlib.sha256()
    copied = 0
    with (
        filesystem.open_input_stream(f"{directory}/{entry.name}") as reader,
        partial.open("xb") as writer,
    ):
        while chunk := reader.read(_CHUNK_BYTES):
            copied += len(chunk)
            if copied > entry.size:
                raise ValueError("Artifact exceeds its advertised size")
            writer.write(chunk)
            digest.update(chunk)
    if copied != entry.size or digest.hexdigest() != entry.sha256:
        raise ValueError("Artifact size or checksum mismatch")
    os.replace(partial, target)


def _relocate(value: object, source: str, destination: str) -> object:
    if isinstance(value, dict):
        return {
            key: _relocate(item, source, destination) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relocate(item, source, destination) for item in value]
    if isinstance(value, str) and (value == source or value.startswith(source + "/")):
        return destination + value[len(source) :]
    return value


def relocate_artifacts(
    bundle: ArtifactBundle, root: Path, *, destination_root: Path | None = None
) -> None:
    """Rebase copied JSON paths and refresh task-sidecar checksum inventories.

    Args:
        bundle: Successfully downloaded publication.
        root: Local directory containing the downloaded inventory's files.
        destination_root: Local replacement for bundle.source_root. Defaults to
            root; replicas use their worker root to rebase sibling artifact paths.
    Raises:
        ValueError: If JSON metadata is malformed or checksum paths are unsafe.
        OSError: If local metadata cannot be read or written.
    """
    checksums = []
    manifests = []
    replacement = str(
        (root if destination_root is None else destination_root).resolve()
    )
    for entry in bundle.files:
        path = root / relative_artifact_path(entry.name)
        if path.suffix != ".json" and path.name != "_SUCCESS":
            continue
        if path.name == "_MANIFEST.json":
            manifests.append(path)
            continue
        if path.name == "checksums.json":
            checksums.append(path)
            continue
        value = orjson.loads(path.read_bytes())
        relocated = _relocate(value, bundle.source_root, replacement)
        if relocated != value:
            path.write_bytes(orjson.dumps(relocated))
    for path in checksums:
        values = orjson.loads(path.read_bytes())
        for key in values:
            if not key.endswith("_sha256"):
                raise ValueError("Invalid sidecar checksum entry")
            sidecar = path.parent / relative_artifact_path(key.removesuffix("_sha256"))
            with sidecar.open("rb") as reader:
                values[key] = hashlib.file_digest(reader, "sha256").hexdigest()
        path.write_bytes(orjson.dumps(values))
    for path in manifests:
        value = orjson.loads(path.read_bytes())
        for entry in value["files"]:
            entry["size_bytes"] = (
                (path.parent / relative_artifact_path(entry["rel_path"])).stat().st_size
            )
        path.write_bytes(orjson.dumps(value))
