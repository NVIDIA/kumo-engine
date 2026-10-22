# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumo_core.sources.cpu import COPY_WORKER_BYTES, memory_share
from kumoapi.data_source import DataSourceType
from pyarrow.fs import FSSpecHandler, PyFileSystem

from .backend import ADLSSessions, canonical_uri
from .native_output import NativeOutputHandler, filesystem_for
from .sdk_filesystem import ADLSFileSystem


class ADLSSinkBackend(ADLSSessions, ObjectStoreSinkBackend):
    source_type = DataSourceType.ADLS

    def _uri(self, path: str) -> str:
        return canonical_uri(path)

    def _filesystem(self, connector, bucket, region):
        filesystem = super()._filesystem(connector, bucket, region)
        if (isinstance(filesystem, PyFileSystem)
                and isinstance(filesystem.handler, FSSpecHandler)
                and isinstance(filesystem.handler.fs, ADLSFileSystem)):
            container, host = bucket.split('@', 1)
            account, suffix = host.split('.', 1)
            # Model reads use this filesystem too; bind output credentials
            # only when a native output stream is actually requested.
            native = partial(filesystem_for, self._principal(connector),
                             account, suffix, container)
            return PyFileSystem(
                NativeOutputHandler(filesystem.handler.fs,
                                    output_factory=native))
        return filesystem

    def _download_file(self, filesystem, source, destination):
        if (isinstance(filesystem, PyFileSystem)
                and isinstance(filesystem.handler, FSSpecHandler)
                and isinstance(filesystem.handler.fs, ADLSFileSystem)):
            with memory_share(COPY_WORKER_BYTES, lambda ceiling: 1):
                filesystem.handler.fs.fetch(source, destination)
        else:
            super()._download_file(filesystem, source, destination)
