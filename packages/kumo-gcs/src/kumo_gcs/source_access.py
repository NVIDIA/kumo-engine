# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from google.auth.credentials import Credentials
from google.cloud import storage
from kumo_core.sources.object_store_access import ObjectStoreAccess
from pyarrow.fs import FSSpecHandler, PyFileSystem

from .download import GCSDownloader
from .errors import translated
from .sdk_filesystem import GCSReadFileSystem


def source_access(
        *, credentials: Credentials, project: str | None = None,
        requester_pays_project: str | None = None) -> ObjectStoreAccess:
    if credentials is None:
        raise TypeError('GCS source access requires explicit credentials.')
    project = project or requester_pays_project
    downloader = GCSDownloader(credentials=credentials, project=project,
                               requester_pays_project=requester_pays_project)

    def fetch(source: str, target: str) -> int | None:
        bucket, separator, key = source.partition('/')
        if not bucket or not separator or not key:
            raise ValueError('GCS source must contain a bucket and object key.')
        with translated(source):
            return downloader.download(bucket, key, target)

    if requester_pays_project is not None:
        filesystem = PyFileSystem(
            FSSpecHandler(
                GCSReadFileSystem(
                    storage.Client(credentials=downloader.credentials,
                                   project=project), requester_pays_project)))
    else:
        filesystem = downloader.filesystem()
    return ObjectStoreAccess(filesystem, fetch)
