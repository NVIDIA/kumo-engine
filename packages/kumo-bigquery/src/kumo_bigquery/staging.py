# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any

from kumo_bigquery import errors
from kumo_core.sources.cpu import download_workers
from kumo_core.sources.errors import PermanentSourceError, TransientSourceError
from kumo_core.sources.transfer import (
    StagedFile,
    fetch_all,
    remove_all,
    remove_quietly,
    staged_fetch,
)
from kumo_core.sources.types import Capability

SCHEME = "gs://"
OBJECT_PATTERN = "part-*.parquet"
PROBE_PATTERN = "probe-*.parquet"
PROBE_PREFIX = "_probe"
PROBE_TIMEOUT_SECONDS = 60
DELETE_BATCH = 100  # JSON API batch limit

EXPORT = ("EXPORT DATA OPTIONS(uri='{uri}', format='PARQUET', overwrite=true) "
          "AS SELECT {projection} FROM {source}")
PROBE = ("EXPORT DATA OPTIONS(uri='{uri}', format='PARQUET', overwrite=true) "
         "AS SELECT 1 AS probe")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagingLocation:
    bucket: str
    prefix: str

    @classmethod
    def parse(cls, uri: str) -> StagingLocation:
        if not uri.startswith(SCHEME):
            raise ValueError(f"a staging uri must start with {SCHEME}: {uri!r}")
        bucket, _, prefix = uri[len(SCHEME):].partition("/")
        if not bucket:
            raise ValueError(f"staging uri {uri!r} names no bucket")
        return cls(bucket=bucket, prefix=prefix.strip("/"))

    def child(self, name: str) -> StagingLocation:
        return StagingLocation(bucket=self.bucket,
                               prefix=f"{self.prefix}/{name}".strip("/"))

    @property
    def directory(self) -> str:
        return f"{self.prefix}/" if self.prefix else ""

    def object_uri(self, pattern: str = OBJECT_PATTERN) -> str:
        """The export target, whose wildcard BigQuery requires."""
        return f"{SCHEME}{self.bucket}/{self.directory}{pattern}"


def export(
    client: Any,
    *,
    location: StagingLocation,
    projection: str,
    source: str,
    job_config: Any,
) -> int:
    job = client.query(
        EXPORT.format(uri=location.object_uri(), projection=projection,
                      source=source), job_config=job_config)
    job.result()
    return int(job.total_bytes_processed or 0)


def probe(
    client: Any,
    storage_client: Any,
    *,
    location: StagingLocation,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> Capability:
    """Probed with a literal rather than the real projection: a view's SQL is
    inlined, and the plan scans its whole input before the write stage is
    refused -- 5.5M rows for an answer this gets unbilled. Nothing raises out
    of here, because no probe may fail a read.
    """
    from google.api_core.exceptions import Forbidden

    target = location.child(f"{PROBE_PREFIX}/{uuid.uuid4().hex}")
    try:
        try:
            client.query(PROBE.format(
                uri=target.object_uri(PROBE_PATTERN))).result(timeout=timeout)
        except Forbidden:
            return Capability.DENIED
        except Exception:
            logger.warning(
                "could not establish whether %s may be exported to, so it "
                "will not be used", target.object_uri(PROBE_PATTERN),
                exc_info=True)
            return Capability.UNKNOWN
        return Capability.ALLOWED
    finally:
        remove_quietly(lambda: clear_prefix(storage_client, target),
                       target.object_uri(PROBE_PATTERN))


def clear_prefix(
    storage_client: Any,
    location: StagingLocation,
    *,
    worker_ceiling: int | None = None,
) -> None:
    from google.api_core.exceptions import NotFound
    from google.cloud.storage.retry import DEFAULT_RETRY

    names = _listed(storage_client, location)
    if not names:
        return
    bucket = storage_client.bucket(location.bucket)
    remove_all([
        partial(DEFAULT_RETRY(_delete), storage_client, bucket,
                names[start:start + DELETE_BATCH])
        for start in range(0, len(names), DELETE_BATCH)
    ], rules=errors.rules(), worker_ceiling=worker_ceiling)
    left = _listed(storage_client, location)
    if left:
        # A batch raises one error, so a tolerated 404 can hide a 403.
        for name in left[:3]:
            with suppress(NotFound):
                bucket.blob(name).delete()
        raise TransientSourceError(
            f"{len(left)} objects were still staged under "
            f"{location.object_uri()} after they were deleted")


def _listed(storage_client: Any, location: StagingLocation) -> list[str]:
    return [
        blob.name for blob in storage_client.list_blobs(
            location.bucket, prefix=location.directory)
    ]


def _delete(storage_client: Any, bucket: Any, names: list[str]) -> None:
    from google.api_core.exceptions import NotFound
    with suppress(NotFound), storage_client.batch():
        for name in names:
            bucket.blob(name).delete()


def download(
    storage_client: Any,
    location: StagingLocation,
    dest: str,
    *,
    worker_ceiling: int | None = None,
) -> tuple[str, ...]:
    """No size is carried from the listing: the fetch is checksummed, and the
    listing describes a generation the fetch does not read."""
    files = tuple(
        StagedFile(listed.name) for listed in storage_client.list_blobs(
            location.bucket, prefix=location.directory))
    if not files:
        raise PermanentSourceError(
            f"the export left nothing under {location.object_uri()!r}")
    bucket = storage_client.bucket(location.bucket)
    return fetch_all(files, dest, fetch=partial(_fetch, bucket),
                     worker_ceiling=worker_ceiling)


def staged_read(
    client: Any,
    storage_client: Any,
    *,
    location: StagingLocation,
    projection: str,
    source: str,
    dest: str,
    worker_ceiling: int | None = None,
    job_config: Any,
) -> tuple[tuple[str, ...], int]:
    def unload() -> int:
        return export(client, location=location, projection=projection,
                      source=source, job_config=job_config)

    def fetch(_: int) -> tuple[str, ...]:
        return download(storage_client, location, dest,
                        worker_ceiling=worker_ceiling)

    scanned, paths = staged_fetch(
        clear=partial(clear_prefix, storage_client, location,
                      worker_ceiling=worker_ceiling), unload=unload,
        fetch=fetch, location=location.object_uri())
    return paths, scanned


def _fetch(bucket: Any, staged: StagedFile, target: str) -> None:
    """Downloaded whole so the client checksums it against the bucket's
    crc32c, which a size and a magic number cannot do. The handle is minted
    fresh because a listed one pins a generation an overwrite may have already
    removed."""
    bucket.blob(staged.path).download_to_filename(target)


def probe_upload(
    storage_client: Any,
    *,
    location: StagingLocation,
) -> Capability:
    """The write direction of :func:`probe`: whether this identity may write,
    list, read and delete objects under ``location``, the four things a staged
    write does there. Nothing raises out of here."""
    from google.api_core.exceptions import Forbidden, NotFound

    target = location.child(f"{PROBE_PREFIX}/{uuid.uuid4().hex}")
    blob = storage_client.bucket(target.bucket).blob(f"{target.directory}probe")
    try:
        blob.upload_from_string(b"")
        list(
            storage_client.list_blobs(target.bucket, prefix=target.directory,
                                      max_results=1))
        blob.reload()
        blob.delete()
    except (Forbidden, NotFound):
        return Capability.DENIED
    except Exception:
        logger.warning(
            "could not establish whether %s may be uploaded to, so it will "
            "not be used", target.object_uri("probe"), exc_info=True)
        return Capability.UNKNOWN
    finally:
        remove_quietly(lambda: clear_prefix(storage_client, target),
                       target.object_uri("probe"))
    return Capability.ALLOWED


def upload(
    storage_client: Any,
    location: StagingLocation,
    files: list[Any],
    *,
    worker_ceiling: int | None = None,
) -> list[str]:
    """``files`` are callables producing a local path, so preparing a file
    and uploading it share a worker. Waits for every upload, as
    :func:`kumo_core.sources.transfer.remove_all` does, so no straggler
    writes after the prefix is cleared."""
    from concurrent.futures import ThreadPoolExecutor

    bucket = storage_client.bucket(location.bucket)

    def put(index: int, prepared: Any) -> str:
        name = f"{location.directory}part-{index:08d}.parquet"
        with prepared() as path:
            bucket.blob(name).upload_from_filename(
                str(path), content_type="application/octet-stream")
        return f"{SCHEME}{location.bucket}/{name}"

    pool = ThreadPoolExecutor(
        min(download_workers(worker_ceiling), max(len(files), 1)))
    try:
        futures = [
            pool.submit(put, index, prepared)
            for index, prepared in enumerate(files)
        ]
        for future in futures:
            if future.exception() is not None:
                pool.shutdown(cancel_futures=True)
                raise future.exception()  # type: ignore[misc]
        return [future.result() for future in futures]
    finally:
        pool.shutdown(cancel_futures=True)


def staged_write(
    storage_client: Any,
    *,
    location: StagingLocation,
    files: list[Any],
    load: Any,
    worker_ceiling: int | None = None,
) -> None:
    """The write direction of :func:`staged_read`: upload, one load job over
    exactly the uploaded objects, and the prefix cleared either way."""
    try:
        clear_prefix(storage_client, location)
        load(
            upload(storage_client, location, files,
                   worker_ceiling=worker_ceiling))
    finally:
        remove_quietly(lambda: clear_prefix(storage_client, location),
                       location.object_uri())


def bucket_location(storage_client: Any,
                    location: StagingLocation) -> str | None:
    """``None`` when the identity may use the bucket's objects but not read
    the bucket itself."""
    from google.api_core.exceptions import Forbidden
    try:
        return storage_client.get_bucket(location.bucket).location
    except Forbidden:
        return None


def colocated(bucket: str, dataset: str) -> bool:
    """Whether a bucket sits in, or inside, the dataset's location."""
    bucket, dataset = bucket.upper(), dataset.upper()
    if bucket == dataset:
        return True
    within = {"US": ("US-", "NAM4"), "EU": ("EUROPE-", "EUR4")}.get(dataset)
    return within is not None and bucket.startswith(within)
