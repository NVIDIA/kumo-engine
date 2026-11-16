# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import os
import threading
import time
import uuid
from contextlib import suppress
from typing import cast

import pyarrow as pa
from google.api_core import exceptions as gcs_exceptions
from google.auth.credentials import (
    AnonymousCredentials,
    Credentials,
    with_scopes_if_required,
)
from google.auth.transport import requests
from google.cloud import storage
from google.cloud.storage._media.requests import _request_helpers
from kumo_core.sources.cpu import cpu_budget
from pyarrow import fs as pafs

logger = logging.getLogger(__name__)

_GCS_DOWNLOAD_MAX_CONCURRENCY_ENV = 'FSUTIL_GCS_DOWNLOAD_MAX_CONCURRENCY'
_GCS_DOWNLOAD_MAX_CONCURRENCY_DEFAULT = 4


def _configured_gcs_download_max_concurrency() -> int:
    raw_value = os.environ.get(_GCS_DOWNLOAD_MAX_CONCURRENCY_ENV)
    if raw_value is None:
        return _GCS_DOWNLOAD_MAX_CONCURRENCY_DEFAULT
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    if value < 1:
        logger.warning(
            '%s must be a positive integer; got %r, using default %d',
            _GCS_DOWNLOAD_MAX_CONCURRENCY_ENV, raw_value,
            _GCS_DOWNLOAD_MAX_CONCURRENCY_DEFAULT)
        return _GCS_DOWNLOAD_MAX_CONCURRENCY_DEFAULT
    return value


_gcs_download_max_concurrency = _configured_gcs_download_max_concurrency()
_gcs_download_semaphore = threading.BoundedSemaphore(
    _gcs_download_max_concurrency)

#: ``google-cloud-storage`` streams a single-shot download through Python in
#: ``_SINGLE_GET_CHUNK_SIZE``-byte pieces, and ships that constant at 8 KiB.
#: Every piece is a Python-level loop iteration holding the GIL, so the chunk
#: size sets how much CPU a download costs and how much of it can overlap.
#: Measured over a 2.55 GB / 484-object GCS tree through ``fsutil.copy``:
#: 8 KiB gives 324 MB/s at 4.04 CPU-s/GB, 1 MiB gives 462 MB/s at 3.76.
#: The library exposes no public knob -- ``Blob.download_to_file`` takes no
#: chunk argument -- so the module constant is the only lever.
#: The download tests assert the attribute still exists, so an upgrade that
#: renames it fails loudly instead of silently restoring 8 KiB.
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_request_helpers._SINGLE_GET_CHUNK_SIZE = _DOWNLOAD_CHUNK_SIZE

# Each single-file copy_files occupies an Arrow I/O worker while transferring;
# 32 matches generic.copy's file-worker count instead of queueing behind Arrow's
# default 8. Historical tuning on an 18.69 GB snapshot measured pool sizes
# 8/32/64/128/192 at 405/2225/2736/3280/1944 MB/s. That curve motivated the
# override ceiling of 128; it is not a benchmark of a contended GKE daemon.
# The default ceiling of 64 limits process-wide impact on pods with no CPU
# limit, where cpu_budget() sees the node. It trades away 16.6% throughput on
# that curve versus 128; FSUTIL_GCS_IO_THREADS=128 explicitly opts into it.
# Never restore a previous pool size: another concurrent copy may still need it.
_GCS_IO_THREADS_ENV = 'FSUTIL_GCS_IO_THREADS'
_GCS_IO_THREADS_PER_CPU = 4
_GCS_IO_THREADS_MIN = 32
_GCS_IO_THREADS_DEFAULT_MAX = 64
_GCS_IO_THREADS_MAX = 128


def gcs_io_threads() -> int:
    raw = os.environ.get(_GCS_IO_THREADS_ENV, '').strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 1:
            return min(value, _GCS_IO_THREADS_MAX)
        logger.warning(
            '%s must be a positive integer; got %r, deriving from '
            'the CPU budget instead', _GCS_IO_THREADS_ENV, raw)
    derived = _GCS_IO_THREADS_PER_CPU * cpu_budget()
    # Pods without a CPU limit can see the whole node's CPU count.
    return max(_GCS_IO_THREADS_MIN, min(_GCS_IO_THREADS_DEFAULT_MAX, derived))


# Match the SDK's 120-second retry deadline. google-cloud-cpp's much longer
# default can hold a worker through a brownout while the restore keeps
# heartbeating, so Temporal's heartbeat timeout would not interrupt the wait.
_GCS_RETRY_SECONDS = 120.0

_GCS_TOKEN_REFRESH_MARGIN = datetime.timedelta(minutes=10)
_gcs_io_pool_lock = threading.Lock()


class _ScopedDownloadFilesystem:
    """Share one token snapshot per scope; renew between file downloads."""
    def __init__(self, credential: Credentials):
        self.credential = credential
        self.lock = threading.Lock()
        # Arrow's stubs omit these public filesystem re-exports.
        self.filesystem: pafs.GcsFileSystem | None = None  # pyright: ignore[reportPrivateImportUsage]
        self.refresh_at = datetime.datetime.min.replace(
            tzinfo=datetime.timezone.utc)

    def get(
        self
    ) -> pafs.GcsFileSystem:  # pyright: ignore[reportPrivateImportUsage]
        with self.lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            if self.filesystem is not None and now < self.refresh_at:
                return self.filesystem
            expiry = self.credential.expiry
            if (not self.credential.token or expiry is None
                    or _utc_expiry(expiry) <= now + _GCS_TOKEN_REFRESH_MARGIN):
                self.credential.refresh(requests.Request())
            # Metadata requests also use this credential. Read expiry first
            # so a concurrent token renewal cannot extend an old token's TTL.
            expiry = self.credential.expiry
            token = self.credential.token
            now = datetime.datetime.now(datetime.timezone.utc)
            if expiry is None or not token:
                raise ValueError('GCS credential has no token or expiry')
            expiry = _utc_expiry(expiry)
            if expiry <= now:
                raise ValueError(
                    'GCS credential refresh returned an expired token')
            filesystem = pafs.GcsFileSystem(  # pyright: ignore[reportPrivateImportUsage]
                access_token=token, credential_token_expiration=expiry,
                retry_time_limit=datetime.timedelta(seconds=_GCS_RETRY_SECONDS))
            margin = _GCS_TOKEN_REFRESH_MARGIN
            # Providers issuing tokens shorter than the margin must not
            # trigger another refresh on every file.
            if expiry - now <= margin:
                margin = (expiry - now) / 2
            self.filesystem = filesystem
            self.refresh_at = expiry - margin
            return filesystem


def _utc_expiry(expiry: datetime.datetime) -> datetime.datetime:
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=datetime.timezone.utc)
    return expiry.astimezone(datetime.timezone.utc)


class GCSDownloader:
    def __init__(self, credentials: Credentials | None = None,
                 project: str | None = None,
                 requester_pays_project: str | None = None):
        self.credentials = (cast(
            Credentials,
            with_scopes_if_required(credentials, scopes=storage.Client.SCOPE))
                            if credentials is not None else None)
        self.project = project
        self.requester_pays_project = requester_pays_project
        self._clients = threading.local()
        self._scoped = (_ScopedDownloadFilesystem(self.credentials)
                        if self.credentials is not None and
                        not isinstance(self.credentials, AnonymousCredentials)
                        else None)
        self._filesystem_lock = threading.Lock()
        self._ambient_filesystem = None
        self._ambient_expires = 0.0

    def _blob(self, bucket: str, key: str):
        if not hasattr(self._clients, 'client'):
            project = self.project
            if project is None and isinstance(self.credentials,
                                              AnonymousCredentials):
                project = ''
            self._clients.client = storage.Client(project=project,
                                                  credentials=self.credentials)
        return self._clients.client.bucket(
            bucket, user_project=self.requester_pays_project).blob(key)

    def filesystem(self):
        if self._scoped is not None:
            return self._scoped.get()
        with self._filesystem_lock:
            if (self._ambient_filesystem is None
                    or time.monotonic() >= self._ambient_expires):
                self._ambient_filesystem = pafs.GcsFileSystem(  # pyright: ignore[reportPrivateImportUsage]
                    anonymous=isinstance(self.credentials,
                                         AnonymousCredentials),
                    retry_time_limit=datetime.timedelta(
                        seconds=_GCS_RETRY_SECONDS))
                self._ambient_expires = time.monotonic() + 600
            return self._ambient_filesystem

    def _requires_sdk_download(self) -> bool:
        return bool(
            os.environ.get('STORAGE_EMULATOR_HOST')
            or self.requester_pays_project
            or getattr(self.credentials, 'quota_project_id', None))

    def object_identity(self, bucket: str, key: str) -> tuple[str, ...]:
        """Fetch current metadata, never the generation pinned by an old Blob."""
        probe = self._blob(bucket, key)
        probe.reload()
        if probe.generation is None:
            raise RuntimeError('GCS object metadata has no generation')
        return (str(probe.generation), str(probe.size))

    def _current_generation(self, bucket: str, key: str):
        probe = self._blob(bucket, key)
        try:
            probe.reload()
        except gcs_exceptions.NotFound:
            return None
        return probe.generation

    def _download_native(self, bucket: str, key: str, target: str) -> None:
        target_threads = gcs_io_threads()
        with _gcs_io_pool_lock:
            if pa.io_thread_count() < target_threads:
                pa.set_io_thread_count(target_threads)
        # Transfers retain their token snapshot; renewal occurs between files.
        source_fs = self.filesystem()
        local = pafs.LocalFileSystem(  # pyright: ignore[reportPrivateImportUsage]
        )
        try:
            pafs.copy_files(f'{bucket}/{key}', target,
                            source_filesystem=source_fs,
                            destination_filesystem=local)
        except (FileNotFoundError, PermissionError):
            raise
        except pa.ArrowCapacityError as error:
            raise ConnectionError(str(error)) from error
        except OSError as error:
            # Preserve local disk errors; only unclassified transport failures
            # should trigger the caller's whole-file retry policy.
            if error.errno is not None:
                raise
            raise ConnectionError(str(error)) from error

    def download(self, bucket: str, key: str, target: str) -> int | None:
        temporary = os.path.join(
            os.path.dirname(target),
            f'.{os.path.basename(target)}.gcsdl-{uuid.uuid4().hex}')
        try:
            blob = self._blob(bucket, key)
            blob.reload()
            expected_size = blob.size
            # Download response headers can overwrite Blob.content_encoding.
            is_gzip_encoded = blob.content_encoding == 'gzip'
            if is_gzip_encoded or self._requires_sdk_download():
                wait_started = time.monotonic()
                with _gcs_download_semaphore:
                    waited = time.monotonic() - wait_started
                    if waited >= 0.1:
                        logger.debug(
                            'GCS download permit acquired after %.3fs (limit=%d)',
                            waited, _gcs_download_max_concurrency)
                    with open(temporary, 'wb') as output:
                        blob.download_to_file(output)
            else:
                expected_generation = blob.generation
                self._download_native(bucket, key, temporary)
                # A fresh Blob avoids rereading the old pinned generation.
                current_generation = self._current_generation(bucket, key)
                if current_generation != expected_generation:
                    raise IOError(
                        f'File gs://{bucket}/{key} was replaced while it was '
                        f'being copied: generation {expected_generation} '
                        f'became {current_generation}.')
            actual_size = os.path.getsize(temporary)
            if not is_gzip_encoded and expected_size != actual_size:
                raise IOError(
                    f'File gs://{bucket}/{key} might be incomplete! Expected '
                    f'{expected_size} bytes, but got {actual_size} bytes.')
            os.replace(temporary, target)
            return actual_size if is_gzip_encoded else None
        except gcs_exceptions.NotFound:
            raise FileNotFoundError(f'gs://{bucket}/{key}') from None
        finally:
            with suppress(OSError):
                os.remove(temporary)
