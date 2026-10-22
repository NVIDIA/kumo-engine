# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumo_core.sources.cpu import COPY_WORKER_BYTES, memory_share
from kumoapi.data_source import DataSourceType
from pyarrow.fs import FSSpecHandler, PyFileSystem

from .backend import ADLSSessions, canonical_uri
from .sdk_filesystem import ADLSFileSystem


class ADLSSinkBackend(ADLSSessions, ObjectStoreSinkBackend):
    source_type = DataSourceType.ADLS

    def _uri(self, path: str) -> str:
        return canonical_uri(path)

    def _upload_file(self, filesystem, source, destination):
        if (isinstance(filesystem, PyFileSystem)
                and isinstance(filesystem.handler, FSSpecHandler)
                and isinstance(filesystem.handler.fs, ADLSFileSystem)):
            filesystem.handler.fs.put_file(source, destination)
        else:
            super()._upload_file(filesystem, source, destination)

    def _download_file(self, filesystem, source, destination):
        if (isinstance(filesystem, PyFileSystem)
                and isinstance(filesystem.handler, FSSpecHandler)
                and isinstance(filesystem.handler.fs, ADLSFileSystem)):
            with memory_share(COPY_WORKER_BYTES, lambda ceiling: 1):
                filesystem.handler.fs.fetch(source, destination)
        else:
            super()._download_file(filesystem, source, destination)
