# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from contextlib import suppress
from functools import partial
from typing import Sequence

from databricks.sdk import WorkspaceClient
from databricks.sql.client import Cursor
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.sources.transfer import (
    DOWNLOAD_CHUNK,
    StagedFile,
    fetch_all,
    remove_all,
)
from kumo_databricks import errors
from kumo_databricks.sql import read_query

STAGE = "INSERT OVERWRITE DIRECTORY '{location}' USING PARQUET {query}"
PARQUET_SUFFIX = ".parquet"


def stage(cursor: Cursor, *, location: str, projection: str,
          source: str) -> None:
    """``INSERT OVERWRITE DIRECTORY`` leaves ``rowcount`` at -1, so the rows
    are counted from the Parquet that arrives."""
    if "'" in location:
        raise ValueError(f"a staging location cannot be quoted: {location!r}")
    cursor.execute(
        STAGE.format(location=location, query=read_query(source, projection)))


def clear_volume(client: WorkspaceClient, path: str, *,
                 worker_ceiling: int | None = None) -> None:
    from databricks.sdk.errors import NotFound
    try:
        entries = list(client.files.list_directory_contents(path))
    except NotFound:
        return
    remove_all([
        partial(_delete, client, entry.path)
        for entry in entries if not entry.is_directory
    ], rules=errors.rules(), worker_ceiling=worker_ceiling)
    for entry in entries:
        if entry.is_directory:
            clear_volume(client, entry.path, worker_ceiling=worker_ceiling)
    try:
        client.files.delete_directory(path)
    except NotFound:
        # A prefix that is not there is what a first read, and a retry after
        # a successful clear, both look like.
        pass


def _delete(client: WorkspaceClient, path: str) -> None:
    from databricks.sdk.errors import NotFound
    with suppress(NotFound):
        client.files.delete(path)


def list_parquet(client: WorkspaceClient, path: str) -> tuple[StagedFile, ...]:
    """The unload also writes ``_SUCCESS`` and commit markers, which are not
    Parquet and would fail to open at ingest."""
    from databricks.sdk.errors import NotFound
    try:
        entries = list(client.files.list_directory_contents(path))
    except NotFound as error:
        raise PermanentSourceError(
            f"databricks staged nothing to {path}: {error}") from error
    files = tuple(
        StagedFile(path=entry.path, size=int(entry.file_size))
        for entry in entries
        if not entry.is_directory and entry.path.endswith(PARQUET_SUFFIX))
    if not files:
        raise PermanentSourceError(f"databricks staged no parquet to {path}")
    return files


def download(
    client: WorkspaceClient,
    files: Sequence[StagedFile],
    dest: str,
    *,
    worker_ceiling: int | None = None,
) -> tuple[str, ...]:
    return fetch_all(files, dest, fetch=partial(_fetch, client),
                     worker_ceiling=worker_ceiling)


def _fetch(client: WorkspaceClient, staged: StagedFile, target: str) -> None:
    """The download response is a plain record holding an open stream, with
    neither a context manager nor a finalizer, so a few hundred files leak
    that many pooled connections unless each is closed here."""
    stream = client.files.download(staged.path).contents
    try:
        with open(target, "wb") as handle:
            shutil.copyfileobj(stream, handle, DOWNLOAD_CHUNK)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
