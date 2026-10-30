# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unity Catalog volumes as a pyarrow filesystem, over the Files API."""

from __future__ import annotations

import io
import posixpath
import queue
import threading
from contextlib import contextmanager
from typing import BinaryIO, Dict, Iterator, Optional
from urllib.parse import quote

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound, PermissionDenied
from kumo_core.chunked_fs import ChunkedFileHandler
from kumo_core.sources.diagnostics import safe_summary
from kumo_databricks.staging import clear_volume
from pyarrow.fs import FileSystem, PyFileSystem

_FILES = '/api/2.0/fs/files'
_BUFFER = 8 << 20


def is_volume_path(path: str) -> bool:
    parts = path.rstrip('/').split('/')
    return (path.startswith('/Volumes/') and len(parts) >= 5
            and all(parts[1:]) and posixpath.normpath(path) == path.rstrip('/'))


def volume_filesystem(client: WorkspaceClient) -> FileSystem:
    return PyFileSystem(VolumeHandler(client))


@contextmanager
def _translated(path: str) -> Iterator[None]:
    """OS errors, unchained; non-SDK text may hold a signed URL."""
    try:
        yield
    except NotFound as error:
        raise FileNotFoundError(f'{path}: {error}') from None
    except PermissionDenied as error:
        raise PermissionError(f'{path}: {error}') from None
    except DatabricksError as error:
        raise OSError(f'{path}: {error}') from None
    except Exception as error:
        raise OSError(f'{path}: {safe_summary(error)}') from None


class VolumeHandler(ChunkedFileHandler):
    part_bytes = 4 << 30  # under the Files API's 5 GiB upload cap
    plain_bytes = 256 << 20

    def __init__(self, client: WorkspaceClient) -> None:
        self._client = client

    def get_type_name(self) -> str:
        return 'databricks-volume'

    def _size(self, path: str) -> Optional[int]:
        path = path.rstrip('/')
        with _translated(path):
            try:
                return int(self._client.files.get_metadata(path).content_length)
            except NotFound:
                return None

    def _list(self, directory: str) -> Dict[str, int]:
        directory = directory.rstrip('/')
        with _translated(directory):
            try:
                entries = list(
                    self._client.files.list_directory_contents(directory))
            except NotFound:
                return {}
        found: Dict[str, int] = {}
        for entry in entries:
            path = entry.path.rstrip('/')
            if entry.is_directory:
                found.update(self._list(path))
            else:
                found[path] = int(entry.file_size)
        return found

    def _put(self, local: str, path: str) -> None:
        with _translated(path), open(local, 'rb') as contents:
            self._client.files.upload(path, contents, overwrite=True)

    def _stream(self, path: str) -> BinaryIO:
        return _Upload(self._client, path)

    def _part(self, path: str, size: int) -> _Part:
        return _Part(self._client, path)

    def _remove(self, path: str, *, tree: bool) -> None:
        path = path.rstrip('/')
        with _translated(path):
            if tree:
                clear_volume(self._client, path)
                return
            try:
                self._client.files.delete(path)
            except NotFound:
                pass


class _Part:
    """Serves consecutive reads from one open-ended ranged download."""
    def __init__(self, client: WorkspaceClient, path: str) -> None:
        self._client, self._path = client, path
        self._stream = None
        self._at = 0

    def read_at(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b''
        if self._stream is None or offset != self._at:
            self._open(offset)
        with _translated(self._path):
            pieces, got = [], 0
            while got < length:
                piece = self._stream.read(min(_BUFFER, length - got))
                if not piece:
                    break
                pieces.append(piece)
                got += len(piece)
        self._at += got
        return b''.join(pieces)

    def _open(self, offset: int) -> None:
        self.close()
        with _translated(self._path):
            response = self._client.api_client.do(
                'GET', f'{_FILES}{quote(self._path)}', raw=True,
                response_headers=['content-range'], headers={
                    'Accept': 'application/octet-stream',
                    'Range': f'bytes={offset}-'
                })
        self._stream = response['contents']
        served = response.get('content-range') or ''
        if not served.startswith(f'bytes {offset}-'):
            self.close()
            raise OSError(f'{self._path}: asked for bytes from {offset}, '
                          f'got {served or "the whole file"}')
        self._stream.set_chunk_size(_BUFFER)
        self._at = offset

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class _Feed(io.RawIOBase):
    def __init__(self) -> None:
        self.pieces: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=4)
        self._pending = b''
        self._done = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending and not self._done:
            piece = self.pieces.get()
            if piece is None:
                self._done = True
            else:
                self._pending = piece
        size = min(len(buffer), len(self._pending))
        buffer[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size


class _Upload(io.RawIOBase):
    """One ``files.upload`` fed from writes; ``close`` commits it."""
    def __init__(self, client: WorkspaceClient, path: str) -> None:
        self._path = path
        self._feed = _Feed()
        self._error: Optional[BaseException] = None
        source = io.BufferedReader(self._feed, _BUFFER)

        def upload() -> None:
            try:
                with _translated(path):
                    client.files.upload(path, source, overwrite=True)
            except BaseException as error:
                self._error = error

        self._thread = threading.Thread(target=upload, daemon=True)
        self._thread.start()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        piece = bytes(data)
        self._put(piece)
        return len(piece)

    def _put(self, piece: Optional[bytes]) -> None:
        while self._thread.is_alive():
            try:
                self._feed.pieces.put(piece, timeout=1)
                return
            except queue.Full:
                continue
        self._thread.join()
        if self._error is not None:
            raise self._error
        if piece is not None:
            raise OSError(f'{self._path}: the upload stopped reading')

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._put(None)
            self._thread.join()
            if self._error is not None:
                raise self._error
        finally:
            super().close()
