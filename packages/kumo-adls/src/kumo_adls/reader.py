# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ADLS readers over borrowed clients, independent of engine identities."""

import hashlib
import json
from typing import Literal

from kumo_core.sources.csv_format import CsvOptions
from kumo_core.sources.identity import SourceVersion
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumo_core.types.enums import ConnectorType
from kumoapi.adls import (
    normalize_adls_endpoint_suffix,
    normalize_adls_filesystem,
    normalize_adls_storage_account,
)
from kumoapi.object_store import normalize_object_store_root_prefix
from pyarrow.fs import FSSpecHandler, PyFileSystem

from .sdk_filesystem import ADLSFileSystem
from .source_access import source_access


class ADLSSourceReader(ObjectStoreSourceReader):
    source_type = ConnectorType.ADLS

    def _identities(self, found, worker_ceiling):
        if isinstance(self.filesystem, PyFileSystem):
            handler = self.filesystem.handler
            adapter = handler.fs if isinstance(handler, FSSpecHandler) else None
            if (isinstance(adapter, ADLSFileSystem) and getattr(
                    self.object_identity, '__self__', None) is adapter
                    and getattr(self.object_identity, '__func__',
                                None) is ADLSFileSystem.object_identity):
                identities = adapter.object_identities(
                    [entry.path for entry in found], root=self.root)
                if identities is not None:
                    return identities
        return super()._identities(found, worker_ceiling)

    def _data_version(self, path, entries):
        # The shared single-file version is only an mtime. Include authority
        # and ETags even for one object, so accounts and same-second rewrites
        # cannot alias the same cached source.
        ordered = sorted(entries, key=lambda entry: entry.path)
        metadata = list(
            zip((entry.path for entry in ordered),
                self._identities(ordered, None)))
        # Discovery and preparation spell the same directory with/without '/'.
        single = (len(ordered) == 1
                  and ordered[0].path.rstrip('/') == path.rstrip('/'))
        return SourceVersion(
            hashlib.sha256(
                json.dumps((path.rstrip('/'), metadata)).encode()).hexdigest(),
            'exact' if single else 'listing')


def source_reader(client, account: str, suffix: str = 'dfs.core.windows.net', *,
                  filesystem: str, prefix: str = '',
                  file_format: Literal['csv', 'parquet'] | None = None,
                  file_suffix: str | None = None,
                  csv_options: CsvOptions | None = None) -> ADLSSourceReader:
    """Build the engine's reader over a borrowed, already-authenticated client.

    The caller owns the client and credential for the reader's entire lifetime.
    Prefix is decoded, relative to the filesystem; empty selects its root.
    """
    account = normalize_adls_storage_account(account)
    suffix = normalize_adls_endpoint_suffix(suffix)
    filesystem = normalize_adls_filesystem(filesystem)
    prefix = normalize_object_store_root_prefix(prefix,
                                                provider_name='Azure Data Lake')
    access = source_access(client, account, suffix)
    authority = f'{filesystem}@{account}.{suffix}'
    root = authority + ('/' + prefix if prefix else '')
    return ADLSSourceReader(access.filesystem, root=root,
                            file_format=file_format, file_suffix=file_suffix,
                            csv_options=csv_options,
                            source_type=ConnectorType.ADLS, fetch=access.fetch,
                            object_identity=access.object_identity)
