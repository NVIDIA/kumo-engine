# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import suppress
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from kumo_core.sources.errors import (
    ReconnectRequiredError,
    TransientSourceError,
)
from kumo_core.sources.object_store_access import ObjectStoreAccess, refreshing
from kumo_s3.session import anonymous_filesystem
from pyarrow import fs
from pyarrow.fs import FileSystem

#: Each chunk is one ranged GET: at 2.5 GB/s, 8 MiB is ~300 requests a
#: second where Arrow's default 1 MiB is ~2,400, against S3's 5,500 per prefix.
_CHUNK = 8 << 20


def _object_identity(source: FileSystem, path: str) -> tuple[str, ...]:
    # Unlike FileInfo, native input-file metadata contains opaque provider
    # tokens. Opening performs HEAD, not a GET or a local checksum scan.
    with source.open_input_file(path) as stream:
        metadata = stream.metadata() or {}
        etag = metadata.get('ETag')
        if not etag:
            raise TransientSourceError(
                f'No S3 object identity was returned for {path}.')
        return (etag.decode(), metadata.get('VersionId',
                                            b'').decode(), str(stream.size()))


def source_access(
    bucket: str,
    *,
    session: Any,
    endpoint_url: str | None = None,
    default_region: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_anonymous: bool = False,
) -> ObjectStoreAccess:
    """Objects are fetched through the same Arrow filesystem that lists and
    reads them: Arrow copies outside the GIL, 12x faster than boto3 at 128
    objects at once. Both re-sign when the session's credentials rotate."""
    import kumo_s3

    options = {
        key: value
        for key, value in (options or {}).items() if key not in
        {'access_key', 'secret_key', 'session_token', 'region', 'anonymous'}
    }
    if endpoint_url:
        options.setdefault('endpoint_override', endpoint_url)
        options.setdefault('force_virtual_addressing', True)
    endpoint = options.get('endpoint_override')
    if endpoint:
        endpoint = str(endpoint)
        if '://' not in endpoint:
            endpoint = f"{options.get('scheme', 'https')}://{endpoint}"

    def signed() -> fs.S3FileSystem | None:
        return kumo_s3.filesystem(bucket, session=session,
                                  endpoint_url=endpoint,
                                  default_region=default_region,
                                  options=options)

    filesystem = signed()
    credentials = session.get_credentials() if filesystem else None
    if filesystem is None:
        if not allow_anonymous:
            raise ReconnectRequiredError(
                'The selected S3 identity has no usable credentials.',
                table=bucket)
        filesystem = anonymous_filesystem(
            bucket, region=default_region or session.region_name,
            options=options)
    lock = threading.Lock()
    current = [
        credentials.get_frozen_credentials() if credentials else None,
        filesystem
    ]

    def refreshed() -> fs.FileSystem:
        """Re-signed when the session's credentials rotate."""
        if credentials is None:
            return filesystem
        frozen = credentials.get_frozen_credentials()
        with lock:
            if frozen != current[0]:
                current[:] = [frozen, signed() or current[1]]
            return current[1]

    def object_identity(path: str) -> tuple[str, ...]:
        return _object_identity(refreshed(), path)

    def fetch(path: str, target: str) -> None:
        source_bucket, separator, key = path.partition('/')
        if source_bucket != bucket or not separator or not key:
            raise ValueError('S3 source path is outside the selected bucket.')
        with NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(target)),
                                prefix='.kumo-s3-', delete=False) as temporary:
            staging = temporary.name
        source = refreshed()
        try:
            before = _object_identity(source, path)

            def verify() -> None:
                try:
                    after = object_identity(path)
                except FileNotFoundError as error:
                    raise TransientSourceError(
                        f'{path} was removed while it was being copied.'
                    ) from error
                if after != before:
                    raise TransientSourceError(
                        f'{path} was replaced while it was being copied.')

            try:
                fs.copy_files(path, staging, source_filesystem=source,
                              destination_filesystem=fs.LocalFileSystem(),
                              chunk_size=_CHUNK)
            except Exception:
                verify()
                raise
            verify()
            if os.path.getsize(staging) != int(before[-1]):
                raise TransientSourceError(
                    f'copied {os.path.getsize(staging)} of {before[-1]} '
                    f'bytes of {path}')
            os.replace(staging, target)
        finally:
            with suppress(OSError):
                os.unlink(staging)

    return ObjectStoreAccess(
        refreshing(refreshed) if credentials else filesystem, fetch,
        object_identity)
