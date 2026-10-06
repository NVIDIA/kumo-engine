# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loading a Delta table from Parquet, once, for both backends.

Databricks has no way to write a table directly: Parquet is uploaded to a
Unity Catalog volume with ``PUT INTO`` and loaded from there with ``COPY
INTO``. That sequence -- the stage it lands in, the file names, the staging
table it is copied through, what ``OVERWRITE`` and ``APPEND`` each mean, and
what is removed afterwards whether or not the write succeeded -- is the
destination's contract, not either caller's, so it lives here.

What a caller lends the writer is only *transport*, and only where the two
genuinely differ:

``execute``
    Runs one statement on whatever session the caller holds. The hosted
    connector reaches a pooled provider connection; a local process reaches
    the one ``databricks.sql`` connection its identity opened.
``run``
    Runs a batch of volume-ingestion commands. Hosted fans these out over a
    pool of short-lived sessions with its own retry policy; a local process
    runs them on the session it already has.
``workers``
    How wide that batch may go. Hosted sizes it from environment limits and
    the largest shard, since ``databricks-sql-connector`` buffers a whole
    ``PUT`` source file; a local process stages one file.

The refusals this module makes on purpose -- an artifact outside the managed
stage, a manifest that does not match what was uploaded, nothing to upload --
are raised as :class:`ArtifactValidationError`, so a caller can translate
*that* type at its own boundary without also relabelling a provider or
filesystem failure that happened to surface as some other ``ValueError``
underneath it.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import pandas as pd
from kumo_databricks import output
from kumo_databricks.errors import ArtifactValidationError
from kumo_databricks.sql import qualified_name, quote_identifier, quote_literal

log = logging.getLogger(__name__)

#: Runs one statement and returns whatever the session answers with.
Execute = Callable[[str], Any]

#: Runs volume-ingestion commands. Keyword-for-keyword the hosted connector's
#: own runner, so it is passed unwrapped and stays the thing its tests patch.
RunVolume = Callable[..., None]

#: How wide a batch may go, given the local files a ``PUT`` batch reads --
#: ``None`` for a ``REMOVE`` batch, which reads none -- and its size.
Workers = Callable[[Sequence[Path] | None, int], int]

#: What a shard may be called inside the stage. Interpolated into a volume
#: path, so it is held to characters that cannot restructure one.
_FILE_PREFIX = re.compile(r'[A-Za-z0-9_-]+')

#: A path holding one of these cannot be safely interpolated into a statement.
_UNSAFE = ('\x00', '\n', '\r')

#: Delta stores a naive timestamp only where the table declares it can, and
#: Parquet written from a tz-naive column is exactly that.
_NTZ = "TBLPROPERTIES ('delta.feature.timestampNtz' = 'supported')"

_NOT_STAGEABLE = ('Databricks table output accepts only local Parquet files '
                  'for direct PUT INTO upload or a Kumo managed-volume stage.')


@dataclass(frozen=True)
class VolumeURI:
    """A Unity Catalog volume path, in either spelling it is written in."""
    catalog: str
    uc_schema: str
    volume: str
    path: str

    def as_path(self) -> str:
        return (f'/Volumes/{self.catalog}/{self.uc_schema}/{self.volume}/'
                f'{self.path}')


def parse_volume_uri(path: str) -> VolumeURI | None:
    """Reads either accepted volume spelling, or answers ``None``."""
    candidate = f'dbfs:{path}' if path.startswith('/Volumes/') else path
    parsed = urlsplit(candidate)
    if parsed.scheme != 'dbfs':
        return None
    parts = parsed.path.split('/', 5)
    if len(parts) != 6:
        return None
    leading, volumes, catalog, uc_schema, volume, rest = parts
    if leading or volumes != 'Volumes':
        return None
    if any('/' in part for part in (catalog, uc_schema, volume)):
        return None
    if rest.startswith('/'):
        return None
    return VolumeURI(catalog, uc_schema, volume, rest)


def require_local_path(artifact_path: str) -> None:
    """Refuses anything a ``PUT`` cannot read from this filesystem."""
    if urlsplit(artifact_path).scheme:
        raise ArtifactValidationError(_NOT_STAGEABLE)


def within_stage(artifact_path: str, catalog: str, schema: str) -> str:
    """Requires a volume path to sit under the connector's own staging root.

    The fence around everything this module writes: a stage names a directory
    that will be uploaded to, copied from and then emptied, so one outside the
    managed volume would let a caller reach a place the connector does not
    own.
    """
    volume = parse_volume_uri(artifact_path)
    if volume is None:
        raise ArtifactValidationError(_NOT_STAGEABLE)
    if (volume.catalog.casefold() != catalog.casefold()
            or volume.uc_schema.casefold() != schema.casefold()
            or volume.volume.casefold() != output.STAGING_VOLUME.casefold()):
        raise ArtifactValidationError(
            'Databricks artifact stage is outside the connector-managed '
            'Unity Catalog volume.')
    stage = volume.as_path().rstrip('/')
    if (any(part in {'.', '..'} for part in stage.split('/'))
            or any(character in stage for character in _UNSAFE)):
        raise ArtifactValidationError(
            'Databricks artifact stage contains an unsafe path component.')
    root = (f'/Volumes/{catalog}/{schema}/{output.STAGING_VOLUME}/'
            f'{output.STAGING_PREFIX}')
    if (stage.casefold() != root.casefold()
            and not stage.casefold().startswith(root.casefold() + '/')):
        raise ArtifactValidationError(
            'Databricks artifact stage is outside the connector-managed '
            'artifact staging prefix.')
    return stage


def manifest(stage: str, staged_files: Sequence[str] | None, *, catalog: str,
             schema: str) -> list[str]:
    """Checks the file list a producer says it already uploaded.

    Fails closed: the caller is claiming files exist that this process never
    wrote, so each is held to the stage it was declared under and to being
    Parquet, and a repeated one is refused rather than copied twice.
    """
    if not staged_files:
        raise ArtifactValidationError(
            'A managed-volume producer stage requires its exact uploaded-file '
            'manifest.')
    prefix = stage.rstrip('/') + '/'
    uploaded = []
    for staged_file in staged_files:
        staged_path = within_stage(staged_file, catalog, schema)
        if (not staged_path.casefold().startswith(prefix.casefold())
                or not staged_path.casefold().endswith('.parquet')):
            raise ArtifactValidationError(
                'A staged Databricks artifact file is outside its declared '
                'managed-volume directory or is not Parquet.')
        uploaded.append(staged_path)
    if len(set(uploaded)) != len(uploaded):
        raise ArtifactValidationError(
            'A managed-volume producer manifest contains duplicate files.')
    return uploaded


def stage_files(run: RunVolume, workers: Workers, files: Sequence[Path], *,
                stage: str, file_prefix: str = 'part',
                prepare_file=None) -> tuple[str, list[str]]:
    """Uploads local Parquet into ``stage`` and says what landed where.

    ``PUT INTO`` is the Databricks-recommended ingestion-partner protocol: the
    warehouse answers with a presigned URL and the driver sends the bytes
    straight to the volume's object storage. A partial upload is cleaned up
    before the failure is re-raised, since the stage is copied from as a
    directory and a stray file would be loaded into the table.
    """
    if not files:
        raise ArtifactValidationError(
            'Databricks table output found no local Parquet files to upload.')
    if not _FILE_PREFIX.fullmatch(file_prefix):
        raise ArtifactValidationError(
            'Databricks managed-volume file prefix contains unsupported '
            'characters.')
    remote_files = [
        f'{stage}/{file_prefix}-{index:08d}.parquet'
        for index in range(len(files))
    ]
    commands = [
        f'PUT {quote_literal(str(local))} INTO {quote_literal(remote)} '
        f'OVERWRITE' for local, remote in zip(files, remote_files, strict=True)
    ]
    common_root = os.path.commonpath([str(file.parent) for file in files])
    try:
        if prepare_file is None:
            run(commands, worker_count=workers(files, len(commands)),
                staging_allowed_local_path=common_root)
        else:
            # Transform/upload one file at a time; do not stage another full
            # copy of the selected output just to add destination metadata.
            for local, remote in zip(files, remote_files, strict=True):
                with prepare_file(str(local)) as prepared:
                    run([
                        f'PUT {quote_literal(str(prepared))} INTO '
                        f'{quote_literal(remote)} OVERWRITE'
                    ], worker_count=1,
                        staging_allowed_local_path=str(Path(prepared).parent))
    except BaseException:
        try:
            cleanup(run, workers, remote_files)
        except BaseException as cleanup_error:
            log.warning(
                'Databricks partial managed-volume upload cleanup failed '
                '(error_type=%s)',
                type(cleanup_error).__name__)
        raise
    return stage, remote_files


def cleanup(run: RunVolume, workers: Workers, paths: Sequence[str]) -> None:
    """Removes staged files, newest first, tolerating ones already gone."""
    if not paths:
        return
    commands = [f'REMOVE {quote_literal(path)}' for path in reversed(paths)]
    run(
        commands,
        worker_count=workers(None, len(commands)),
        # `databricks-sql-connector` requires an allowed-local-path value for
        # every volume ingestion command, including REMOVE, even though
        # REMOVE never reads a local file.
        staging_allowed_local_path=os.devnull,
        ignore_missing=True,
    )


def load(
    *,
    execute: Execute,
    run: RunVolume,
    workers: Workers,
    catalog: str,
    schema: str,
    table: str,
    artifact_path: str,
    resolve_local: Callable[[str], Sequence[Path]],
    staged_files: Sequence[str] | None = None,
    replace: bool = False,
    file_prefix: str = 'part',
    prepare_file=None,
) -> None:
    """Writes one Parquet artifact into ``catalog.schema.table``.

    Two entry points, and the destination cannot tell them apart: a local
    Parquet file or dataset, which is uploaded here, or a managed-volume
    directory a distributed producer has already filled, named alongside the
    exact manifest it wrote. S3, ADLS, arbitrary volumes and inline cloud
    credentials are not supported, deliberately.

    Loaded through a staging table rather than into the destination directly,
    so a ``COPY INTO`` that fails part-way leaves the destination as it was;
    the swap at the end is the one statement a reader can observe.

    Everything the artifact is addressed by is checked before a single
    statement runs, because a destination this process may not write to must
    cost nothing to refuse.
    """
    destination_table = qualified_name(schema, table)
    staging_name = f'{table}__kumo_staging_{uuid.uuid4().hex[:16]}'
    staging_table = qualified_name(schema, staging_name)

    volume = parse_volume_uri(artifact_path)
    warehouse_artifact_path = artifact_path
    uploaded_files: list[str] = []
    local_files: Sequence[Path] | None = None
    if volume is None:
        require_local_path(artifact_path)
        local_files = resolve_local(artifact_path)
    else:
        warehouse_artifact_path = within_stage(artifact_path, catalog, schema)
        uploaded_files = manifest(warehouse_artifact_path, staged_files,
                                  catalog=catalog, schema=schema)

    staging_table_touched = False
    write_succeeded = False
    try:
        execute(f'describe schema {quote_identifier(schema)}')
        output.ensure_staging_volume(execute, catalog, schema)

        if volume is None:
            assert local_files is not None
            warehouse_artifact_path, uploaded_files = stage_files(
                run, workers, local_files,
                stage=output.staging_stage(catalog,
                                           schema), file_prefix=file_prefix,
                **({
                    'prepare_file': prepare_file
                } if prepare_file is not None else {}))

        staging_table_touched = True
        execute(f'DROP TABLE IF EXISTS {staging_table}')
        execute(f'CREATE TABLE IF NOT EXISTS {staging_table}')
        execute(f'ALTER TABLE {staging_table} SET {_NTZ}')
        execute(f'COPY INTO {staging_table} '
                f'FROM {quote_literal(warehouse_artifact_path)} '
                f"FILEFORMAT = PARQUET COPY_OPTIONS ('mergeSchema' = 'true')")
        if replace:
            execute(f'CREATE OR REPLACE TABLE {destination_table} {_NTZ} '
                    f'AS SELECT * FROM {staging_table}')
        else:
            execute(f'CREATE TABLE IF NOT EXISTS {destination_table} {_NTZ} '
                    f'AS SELECT * FROM {staging_table} WHERE 1 = 0')
            execute(f'ALTER TABLE {destination_table} SET {_NTZ}')
            execute(f'INSERT INTO {destination_table} BY NAME '
                    f'SELECT * FROM {staging_table}')
        log.info("Data written to '%s'.", destination_table)
        write_succeeded = True
    finally:
        if staging_table_touched:
            with suppress(BaseException):
                execute(f'DROP TABLE IF EXISTS {staging_table}')
        # A producer's own stage is left alone unless the load worked: it
        # owns those files, and a retry has to be able to read them again.
        if volume is None or write_succeeded:
            with suppress(BaseException):
                cleanup(run, workers, uploaded_files)


PARQUET_OPTIONS = dict(coerce_timestamps='us', allow_truncated_timestamps=True)


def to_parquet(frame: pd.DataFrame, path: str) -> None:
    """Writes a frame as the Parquet ``COPY INTO`` will accept.

    Timestamps are coerced to microseconds because ``COPY INTO`` refuses
    nanoseconds outright -- ``[PARQUET_TYPE_ILLEGAL] Illegal Parquet type:
    INT64 (TIMESTAMP(NANOS,false))`` -- and pandas keeps a datetime column in
    them. A Delta ``TIMESTAMP`` holds microseconds, so nothing the destination
    could have kept is lost; truncation is allowed for that reason rather than
    tolerated. A property of what Delta accepts, so it belongs to the write
    and not to whichever process produced the frame.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path,
                   **PARQUET_OPTIONS)
