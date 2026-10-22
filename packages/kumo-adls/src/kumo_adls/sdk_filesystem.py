# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import posixpath
import threading
import weakref
from collections import OrderedDict
from contextlib import suppress
from tempfile import NamedTemporaryFile

from azure.core import MatchConditions
from azure.core.exceptions import ResourceExistsError
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from kumo_core.sources.cpu import COPY_WORKER_BYTES, memory_share
from kumo_core.sources.errors import (
    SourceAccessDeniedError,
    TransientSourceError,
)
from kumo_core.sources.layout import MAX_SCANNED_ENTRIES
from kumoapi.adls import normalize_adls_filesystem

from .errors import translated
from .range_cache import MAX_ENTRY_BYTES, RANGE_BYTES
from .transfer import CHUNK_BYTES, concurrency


def _identity(etag, size):
    # List Paths may omit the quotes present on HEAD response ETags. Tokens
    # remain opaque and case-sensitive; downloads keep their original ETag.
    if isinstance(etag, str) and len(etag) >= 2 and etag[0] == etag[-1] == '"':
        etag = etag[1:-1]
    if not etag or not isinstance(etag,
                                  str) or type(size) is not int or size < 0:
        return None
    return etag, str(size)


def _is_directory(properties):
    # Azure preserves metadata spelling; HNS can return Hdi_isfolder too.
    return any(name.casefold() == 'hdi_isfolder' and value.casefold() == 'true'
               for name, value in (properties.metadata or {}).items())


class _ReadFile(AbstractBufferedFile):
    def __init__(self, filesystem, path, client, properties, **kwargs):
        self.client = client
        self.etag = properties.etag
        super().__init__(filesystem, path, mode='rb', size=properties.size,
                         **kwargs)

    def _fetch_range(self, start, end):
        # fsspec/Arrow may request past EOF (CSV sniffing and readall). The
        # fresh open properties pin size together with the conditional ETag.
        end = min(end, self.size)
        if end <= start:
            return b''
        length = end - start
        key = None
        if length <= MAX_ENTRY_BYTES and isinstance(self.etag,
                                                    str) and self.etag:
            key = (self.fs._range_namespace, self.path, self.etag, start, end)
            cached = RANGE_BYTES.get(key)
            if cached is not None:
                return cached
        with concurrency(length) as workers, translated():
            data = self.client.download_file(
                offset=start, length=length, etag=self.etag,
                match_condition=MatchConditions.IfNotModified,
                max_concurrency=workers).readall()
        if len(data) != length:
            raise TransientSourceError('ADLS download range was truncated')
        if key is not None:
            RANGE_BYTES.put(key, data)
        return data


class _WriteFile(AbstractBufferedFile):
    """Abort upload failures and exceptions in a direct fsspec context.

    Arrow's PythonFile wrapper calls close without forwarding caller errors.
    That path can flush partial output; this is not a transactional writer.
    """
    def __init__(self, filesystem, path, client, **kwargs):
        self.client = client
        self._creation_etag = None
        super().__init__(filesystem, path, mode='wb', **kwargs)

    def _initiate_upload(self):
        with translated():
            self._creation_etag = self.client.create_file().get('etag')
            if not self._creation_etag:
                raise TransientSourceError(
                    'ADLS upload returned no object identity')

    def _upload_chunk(self, final=False):
        data = self.buffer.getvalue()
        try:
            with translated():
                if data:
                    self.client.append_data(data, offset=self.offset,
                                            length=len(data))
                if final:
                    self.client.flush_data(self.offset + len(data))
        except Exception:
            self.discard()
            raise

    def discard(self):
        # Prevent close/__del__ from retrying a failed append or flush.
        if self.closed:
            return
        self.closed = True
        if not self._creation_etag:
            return
        try:
            # Append is unfenced: its response cannot prove we still own the
            # path. Delete only the original creation, never a replacement.
            self.client.delete_file(
                etag=self._creation_etag,
                match_condition=MatchConditions.IfNotModified)
        except Exception:
            # Cleanup must preserve the original upload failure.
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.discard()
        else:
            self.close()


class ADLSFileSystem(AbstractFileSystem):
    """SDK-backed Arrow access with account-qualified virtual paths.

    Virtual paths retain the full ADLS authority so shared versioning and URI
    reconstruction distinguish accounts. Azure clients receive filesystem/path.
    """
    protocol = 'kumo-adls'
    cachable = False

    def __init__(self, client, account: str, suffix: str):
        super().__init__()
        self._range_namespace = object()
        weakref.finalize(self, RANGE_BYTES.clear_namespace,
                         self._range_namespace)
        self.client = client
        self.host = f'{account}.{suffix}'
        self._filesystems = OrderedDict()
        self._filesystems_lock = threading.Lock()

    def _address(self, path):
        path = path.strip('/')
        authority, _, key = path.partition('/')
        filesystem, separator, host = authority.partition('@')
        if not separator or host != self.host:
            raise ValueError(
                'ADLS path does not match the bound storage account')
        normalize_adls_filesystem(filesystem)
        with self._filesystems_lock:
            if filesystem not in self._filesystems:
                self._filesystems[
                    filesystem] = self.client.get_file_system_client(filesystem)
                if len(self._filesystems) > 32:
                    self._filesystems.popitem(last=False)
            else:
                self._filesystems.move_to_end(filesystem)
            client = self._filesystems[filesystem]
        return client, key, authority

    @staticmethod
    def _entry(authority, item):
        directory = item.is_directory
        return {
            'name': f'{authority}/{item.name}'.rstrip('/'),
            'size': 0 if directory else item.content_length,
            'type': 'directory' if directory else 'file',
            'mtime': item.last_modified
        }

    def info(self, path, **kwargs):
        filesystem, key, authority = self._address(path)
        with translated():
            if not key:
                filesystem.get_file_system_properties()
                return {'name': authority, 'size': 0, 'type': 'directory'}
            properties = filesystem.get_file_client(key).get_file_properties()
            directory = _is_directory(properties)
            return {
                'name': f'{authority}/{key}',
                'size': 0 if directory else properties.size,
                'type': 'directory' if directory else 'file',
                'mtime': properties.last_modified,
            }

    def ls(self, path, detail=True, **kwargs):
        info = self.info(path)
        if info['type'] == 'file':
            entries = [info]
        else:
            filesystem, key, authority = self._address(path)
            with translated():
                entries = [
                    self._entry(authority,
                                item) for item in filesystem.get_paths(
                                    path=key or None, recursive=False)
                ]
        return entries if detail else [entry['name'] for entry in entries]

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        info = self.info(path)
        if info['type'] == 'file':
            entries = [info]
        else:
            filesystem, key, authority = self._address(path)
            with translated():
                entries = []
                for item in filesystem.get_paths(path=key or None,
                                                 recursive=maxdepth != 1):
                    relative = item.name[len(key.rstrip('/')) +
                                         1:] if key else item.name
                    if maxdepth is not None and len(
                            relative.split('/')) > maxdepth:
                        continue
                    if withdirs or not item.is_directory:
                        entries.append(self._entry(authority, item))
            if withdirs:
                entries.append(info)
        found = {entry['name']: entry for entry in entries}
        return found if detail else sorted(found)

    def mkdir(self, path, create_parents=True, **kwargs):
        filesystem, key, _ = self._address(path)
        if not key:
            # A connector never creates a storage account or filesystem.
            with translated():
                filesystem.get_file_system_properties()
            return
        parts = key.split('/')
        keys = ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
        for name in keys if create_parents else [key]:
            with translated():
                try:
                    filesystem.get_directory_client(name).create_directory()
                except ResourceExistsError:
                    pass

    def makedirs(self, path, exist_ok=False):
        if not exist_ok and self.exists(path):
            raise FileExistsError(path)
        self.mkdir(path)

    def rmdir(self, path):
        filesystem, key, _ = self._address(path)
        if not key:
            raise ValueError('Deleting an ADLS filesystem is not supported')
        with translated():
            filesystem.get_directory_client(key).delete_directory()

    def rm_file(self, path):
        filesystem, key, _ = self._address(path)
        with translated():
            filesystem.get_file_client(key).delete_file()

    def _open(self, path, mode='rb', block_size=None, cache_options=None,
              **kwargs):
        filesystem, key, _ = self._address(path)
        client = filesystem.get_file_client(key)
        if mode == 'rb':
            with translated():
                properties = client.get_file_properties()
            return _ReadFile(self, path, client, properties,
                             block_size=block_size, cache_options=cache_options,
                             cache_type=kwargs.get('cache_type', 'none'))
        if mode == 'wb':
            return _WriteFile(self, path, client, block_size=block_size)
        raise NotImplementedError(
            'ADLS supports binary reads and overwrite writes')

    def object_identities(self, paths, *, root=''):
        """Fresh, bounded listing identities, or None to use per-file HEADs.

        No metadata outlives this call. Listing is an optimization: narrower
        file grants and incomplete list metadata retain the HEAD path.
        """
        if len(paths) < 2 or len(set(paths)) != len(paths):
            return None
        groups = {}
        for path in paths:
            authority, _, key = path.partition('/')
            groups.setdefault(authority, []).append((path, key))
        identities = {}
        # Count unrelated entries too, so sibling-heavy roots cannot turn a
        # metadata optimization into an unbounded recursive scan.
        limit = min(MAX_SCANNED_ENTRIES, max(1024, 4 * len(paths)))
        scanned = 0
        for authority, selected in groups.items():
            if len(selected) < 2:
                return None
            prefix = posixpath.commonpath(
                [posixpath.dirname(key) for _, key in selected])
            if not prefix:
                return None
            root_authority, _, root_key = root.partition('/')
            outside_prefix = (bool(root_key) and prefix != root_key and
                              not prefix.startswith(root_key.rstrip('/') + '/'))
            outside_root = (bool(root)
                            and (authority != root_authority or outside_prefix))
            if outside_root:
                return None
            filesystem, _, _ = self._address(f'{authority}/{prefix}')
            wanted = {key: path for path, key in selected}
            try:
                with translated():
                    for item in filesystem.get_paths(path=prefix,
                                                     recursive=True):
                        scanned += 1
                        if scanned > limit:
                            return None
                        path = wanted.get(item.name)
                        if path is None:
                            continue
                        identity = _identity(item.etag, item.content_length)
                        if item.is_directory or identity is None or path in identities:
                            return None
                        identities[path] = identity
            except (SourceAccessDeniedError, FileNotFoundError,
                    NotImplementedError):
                return None
        if len(identities) != len(paths):
            return None
        return [identities[path] for path in paths]

    def object_identity(self, path):
        filesystem, key, _ = self._address(path)
        with translated():
            properties = filesystem.get_file_client(key).get_file_properties()
        if _is_directory(properties) or not properties.etag:
            raise ValueError('ADLS source must be a versioned file')
        identity = _identity(properties.etag, properties.size)
        if identity is None:
            raise ValueError('ADLS source must have a complete object identity')
        return identity

    def fetch(self, source, destination):
        filesystem, key, _ = self._address(source)
        client = filesystem.get_file_client(key)
        with translated():
            properties = client.get_file_properties()
            target = NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(destination)),
                prefix='.kumo-adls-', delete=False)
            staging = target.name
            try:
                with target:
                    with concurrency(properties.size, ranged=True):
                        written = client.download_file(
                            **({
                                'length': properties.size,
                                'offset': 0
                            } if properties.size else {}), etag=properties.etag,
                            match_condition=MatchConditions.IfNotModified,
                            max_concurrency=1).readinto(target)
                        # The SDK can report the requested size after a short
                        # write, so independently verify the stream position.
                        position = target.tell()
                        if written != properties.size or position != properties.size:
                            raise TransientSourceError(
                                'ADLS download was truncated')
                    target.flush()
                    if os.path.getsize(staging) != properties.size:
                        raise TransientSourceError(
                            'ADLS download was truncated')
                # Close before promotion or cleanup, including on Windows.
                os.replace(staging, destination)
            finally:
                with suppress(OSError):
                    os.unlink(staging)
        return None

    def put_file(self, source, destination, **kwargs):
        filesystem, key, _ = self._address(destination)
        client = filesystem.get_file_client(key)
        with memory_share(COPY_WORKER_BYTES, lambda ceiling: 1), \
                open(source, 'rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            created_etag = None
            try:
                with concurrency(size, upload=True) as workers, translated():
                    created_etag = client.create_file().get('etag')
                    if not created_etag:
                        raise TransientSourceError(
                            'ADLS upload returned no object identity')
                    if size:
                        client.upload_data(
                            stream, length=size, overwrite=False,
                            etag=created_etag,
                            match_condition=MatchConditions.IfNotModified,
                            chunk_size=CHUNK_BYTES, max_concurrency=workers)
            except Exception:
                # Never delete a replacement created by another writer.
                if created_etag:
                    with suppress(Exception):
                        client.delete_file(
                            etag=created_etag,
                            match_condition=MatchConditions.IfNotModified)
                raise
