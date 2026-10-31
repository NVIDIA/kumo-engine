# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream immutable task extensions to Flight sampling workers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
from pyarrow import flight

from kumo_core.diskgraph.sampling_protocol import (
    ENVELOPE_SCHEMA,
    PUSH_VERSION,
    TASK_CHUNK_BYTES,
    encode_frame,
)


def task_files(root: Path) -> tuple[str, list[tuple[str, int]]]:
    """Hash task files in canonical order using bounded reads.

    Args:
        root: Completed, immutable task-extension directory.
    Returns:
        Content digest and relative file names with byte lengths.
    Raises:
        ValueError: If the extension contains symlinks or no files.
        OSError: If the directory cannot be read.
    """
    digest = hashlib.sha256()
    files = []
    for path in sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError("Task extensions must not contain symlinks")
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append((name, size))
        digest.update(orjson.dumps([name, size]))
        with path.open("rb") as source:
            while data := source.read(TASK_CHUNK_BYTES):
                digest.update(data)
    if not files:
        raise ValueError("Task extension contains no files")
    return digest.hexdigest(), files


def upload_task(
    client: flight.FlightClient,
    root: Path,
    extension_ref: str,
    identity: tuple[str, list[tuple[str, int]]],
) -> None:
    """Publish a task once on a replica, verifying its content digest.

    Args:
        client: Replica's reusable Flight connection.
        root: Immutable trainer-local task directory.
        extension_ref: Relative task extension below the replica graph.
        identity: Digest and file inventory from :func:`task_files`.
    Raises:
        ValueError: If the replica does not confirm the transferred identity.
        flight.FlightError: If preparation or transfer fails.
        OSError: If a task file cannot be read.
    """
    key, files = identity
    request = {"extension_ref": extension_ref, "sha256": key}
    results = list(
        client.do_action(flight.Action("task_status", orjson.dumps(request)))
    )
    if len(results) != 1:
        raise ValueError("Invalid task preparation status")
    if results[0].body.to_pybytes() == key.encode():
        return
    descriptor = flight.FlightDescriptor.for_command(
        orjson.dumps({"kind": "prepare_task", "version": PUSH_VERSION, **request})
    )
    writer, reader = client.do_put(descriptor, ENVELOPE_SCHEMA)
    with writer:
        for name, size in files:
            writer.write_batch(
                encode_frame({"kind": "file", "path": name, "size": size}, [])
            )
            with (root / name).open("rb") as source:
                while data := source.read(TASK_CHUNK_BYTES):
                    writer.write_batch(encode_frame({"kind": "data"}, [data]))
        writer.write_batch(encode_frame({"kind": "end"}, []))
        writer.done_writing()
        reply = reader.read()
        if reply is None or reply.to_pybytes() != key.encode():
            raise ValueError("Task preparation checksum was not acknowledged")
