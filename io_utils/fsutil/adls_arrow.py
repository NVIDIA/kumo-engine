# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random-access (range-read) filesystem adapter for ADLS Gen2 parquet reads.

PyArrow cannot read ADLS natively: its C++ ``AzureFileSystem`` consumes
account keys or its own default-credential chain, while identity-backed
connectors carry a federated Python credential object that never leaves the
Azure SDK. Parquet reads therefore used to download whole files to a temp
directory just to open a footer. This adapter exposes the already-
authenticated DataLake clients as an fsspec filesystem, which PyArrow wraps
via ``pyarrow.fs.PyFileSystem(FSSpecHandler(...))`` so parquet metadata and
bounded samples become byte-range reads instead of full-file copies.

The container-scoped client is captured eagerly at construction time, on the
request thread: PyArrow issues reads from its own IO threads, which must not
re-resolve the request-local (``ContextVar``) credential scope.
"""
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.filedatalake import FileSystemClient
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem

from io_utils.fsutil.uri import ADLSURI, ParseURI


class ADLSRangeReadFile(AbstractBufferedFile):
    """Read-only buffered file whose cache misses become ADLS range reads."""
    def __init__(self, fs: 'ADLSArrowFileSystem', path: str, mode: str = 'rb',
                 **kwargs):
        if mode != 'rb':
            raise NotImplementedError(
                'ADLS arrow filesystem supports read-only access')
        self._file_client = fs.fs_client.get_file_client(path)
        super().__init__(fs, path, mode=mode, **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        if end <= start:
            return b''
        return self._file_client.download_file(offset=start,
                                               length=end - start).readall()


class ADLSArrowFileSystem(AbstractFileSystem):
    """fsspec view of one ADLS container through an eagerly-bound client."""
    protocol = 'kumo-adls'
    # Instances wrap a request-scoped credentialed client; fsspec's instance
    # cache would leak one request's identity into the next.
    cachable = False

    def __init__(self, fs_client: FileSystemClient, **kwargs):
        super().__init__(**kwargs)
        self.fs_client = fs_client

    @classmethod
    def from_uri(cls, uri: str) -> 'ADLSArrowFileSystem':
        from io_utils.fsutil.impl_adls import threadsafe_adls
        parsed = ParseURI(uri)
        if not isinstance(parsed, ADLSURI):
            raise ValueError(f'Not an ADLS URI: {uri}')
        client = threadsafe_adls.data_lake_client(
            parsed.storage_account,
            parsed.endpoint_suffix).get_file_system_client(parsed.container)
        return cls(client)

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        return str(path).lstrip('/')

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        if not path:
            return {'name': '', 'size': 0, 'type': 'directory'}
        try:
            props = self.fs_client.get_file_client(path).get_file_properties()
        except ResourceNotFoundError as error:
            raise FileNotFoundError(path) from error
        if str((props.metadata or {}).get('hdi_isfolder')).lower() == 'true':
            return {'name': path, 'size': 0, 'type': 'directory'}
        return {'name': path, 'size': props.size, 'type': 'file'}

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        info = self.info(path)
        if info['type'] == 'file':
            entries = [info]
        else:
            try:
                entries = [{
                    'name':
                    str(child.name),
                    'size':
                    child.content_length or 0,
                    'type':
                    'directory' if child.is_directory else 'file',
                }
                           for child in self.fs_client.get_paths(
                               path=path or None, recursive=False)]
            except ResourceNotFoundError as error:
                raise FileNotFoundError(path) from error
        return entries if detail else [entry['name'] for entry in entries]

    def _open(self, path, mode='rb', block_size=None, autocommit=True,
              cache_options=None, **kwargs):
        return ADLSRangeReadFile(self, self._strip_protocol(path), mode=mode,
                                 block_size=block_size,
                                 cache_options=cache_options)
