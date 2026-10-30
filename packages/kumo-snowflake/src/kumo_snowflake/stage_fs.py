# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A pyarrow filesystem over a Snowflake named internal stage."""

import logging
import posixpath
import shutil
import tempfile
import threading
import uuid
import weakref
from pathlib import Path
from typing import Dict, Optional, Tuple

from kumo_core.chunked_fs import ChunkedFileHandler, spool_root
from kumo_core.sources.diagnostics import safe_summary
from kumo_snowflake.artifacts import (
    _split_stage_location,
    _stage_join,
    _StageFiles,
)
from kumo_snowflake.session import SnowflakeSession, _rows
from kumo_snowflake.staging import (
    DOWNLOAD_TIMEOUT,
    _download_session,
    presigned_urls,
)
from pyarrow.fs import PyFileSystem

logger = logging.getLogger(__name__)

_PROBE = b'kumo-stage-range-probe\n'
_RANGED, _GET, _ENCRYPTED = 'ranged', 'get', 'encrypted'
_read_modes: Dict[Tuple[Optional[str], str], str] = {}
_read_modes_lock = threading.Lock()


def _pattern(relative: str) -> str:
    # PATTERN sees the stage's internal path, so anchor the suffix.
    escaped = ''.join(f'[{c}]' if c in '.$+' else c for c in relative)
    return f'.*/{escaped}'


class StageFileSystemHandler(ChunkedFileHandler):
    # One object up to this size, so COPY INTO reads it.
    part_bytes = 4 << 30

    def __init__(self, session: SnowflakeSession, root: str) -> None:
        self._stage = _split_stage_location(root)[0]
        self._root = root
        self._session = session
        self._files = _StageFiles(session, parallel=1)
        self._urls: Dict[str, str] = {}
        self._http = threading.local()
        self._downloads: Dict[str, Tuple[threading.Lock, list]] = {}
        self._downloads_lock = threading.Lock()
        self._download_dir: Optional[str] = None

    def get_type_name(self):
        return 'snowflake-stage'

    def _relative(self, path: str) -> str:
        return path[len(self._stage) + 1:]

    def _size(self, path: str) -> Optional[int]:
        found = self._files._list(path, tree=False).get('')
        return None if found is None else found.size

    def _list(self, directory: str) -> Dict[str, int]:
        return {
            f'{directory}/{relative}': obj.size
            for relative, obj in self._files._list(directory).items()
        }

    def _put(self, local: str, path: str) -> None:
        _split_stage_location(path)
        self._files._put(Path(local), posixpath.dirname(path))

    def _remove(self, path: str, *, tree: bool) -> None:
        if tree:
            self._files.remove(path)
            return
        parent = posixpath.dirname(path)
        _rows(
            self._session, f'REMOVE {_stage_join(parent, directory=True)} '
            f"PATTERN = '{_pattern(self._relative(path))}'")
        if self._size(path) is not None:
            raise OSError(f'REMOVE left {path} in place.')

    def _forget(self, path: str) -> None:
        super()._forget(path)
        path = path.rstrip('/')
        with self._downloads_lock:
            for name in list(self._downloads):
                if name == path or name.startswith(path + '/'):
                    del self._downloads[name]

    def _part(self, path: str, size: int):
        mode = self._read_mode()
        if mode == _ENCRYPTED:
            raise OSError(
                f'Snowflake stage {self._stage} encrypts files client-side '
                "(SNOWFLAKE_FULL), so LIST reports encrypted sizes and kumo "
                'cannot read its files back. Use a stage created with '
                "ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE').")
        if mode == _RANGED:
            return _RangedPart(self, path)
        return _DownloadedPart(self, path, size)

    def _read_mode(self) -> str:
        key = (getattr(self._session.connection, 'account', None), self._stage)
        with _read_modes_lock:
            if key not in _read_modes:
                _read_modes[key] = self._probe()
            return _read_modes[key]

    def _probe(self) -> str:
        root = _stage_join(self._stage, '.kumo-preflight', uuid.uuid4().hex)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                probe = Path(tmp) / 'probe'
                probe.write_bytes(_PROBE)
                self._files._put(probe, root)
        except Exception as error:
            logger.warning(
                'Could not verify ranged reads on Snowflake stage %s (%s; '
                'the role may lack WRITE), so each file is downloaded whole.',
                self._stage, safe_summary(error))
            return _GET
        try:
            if self._size(f'{root}/probe') != len(_PROBE):
                return _ENCRYPTED
            ranged = self._get(f'{root}/probe', 1, 4) == _PROBE[1:5]
        except OSError:
            ranged = False
        finally:
            try:
                self._files.remove(root)
            except Exception as error:
                logger.warning('Could not remove the range probe: %s',
                               safe_summary(error))
        if not ranged:
            logger.warning(
                'Snowflake stage %s does not serve byte ranges of plaintext '
                'over presigned URLs, so each file is downloaded whole. '
                "Create artifact stages with ENCRYPTION = (TYPE = "
                "'SNOWFLAKE_SSE') for ranged reads.", self._stage)
        return _RANGED if ranged else _GET

    def _url(self, path: str) -> str:
        url = self._urls.get(path)
        if url is None:
            relative = self._relative(path)
            cursor = self._session.connection.cursor()
            try:
                url = presigned_urls(cursor, self._stage[1:],
                                     [relative])[relative]
            finally:
                cursor.close()
            self._urls[path] = url
        return url

    def _get(self, path: str, offset: int, length: int) -> bytes:
        import requests
        if length == 0:
            return b''
        http = getattr(self._http, 'session', None)
        if http is None:
            http = self._http.session = _download_session()
        try:
            response = http.get(
                self._url(path), timeout=DOWNLOAD_TIMEOUT,
                headers={'Range': f'bytes={offset}-{offset + length - 1}'})
            response.raise_for_status()
        except requests.RequestException as error:
            self._urls.pop(path, None)
            status = getattr(getattr(error, 'response', None), 'status_code',
                             None)
            raise OSError(f'Reading {path} failed: '
                          f'{safe_summary(error, status=status)}') from None
        if (response.status_code != 206 or not response.headers.get(
                'Content-Range', '').startswith(f'bytes {offset}-')):
            raise OSError(f'Reading {path} returned no byte range '
                          f'(status={response.status_code}).')
        return response.content

    def _downloaded(self, path: str, size: int) -> Path:
        with self._downloads_lock:
            if self._download_dir is None:
                self._download_dir = tempfile.mkdtemp(prefix='kumo-stage-get-',
                                                      dir=spool_root(self))
                weakref.finalize(self, shutil.rmtree, self._download_dir, True)
            lock, local = self._downloads.setdefault(path,
                                                     (threading.Lock(), []))
        with lock:
            if not local:
                target = Path(self._download_dir) / uuid.uuid4().hex / (
                    posixpath.basename(path))
                target.parent.mkdir()
                parent = posixpath.dirname(path)
                self._session.file.get(_stage_join(parent, directory=True),
                                       str(target.parent), parallel=1,
                                       pattern=_pattern(self._relative(path)))
                if not target.is_file() or target.stat().st_size != size:
                    shutil.rmtree(target.parent, ignore_errors=True)
                    raise OSError(f'Downloading {path} did not produce its '
                                  f'{size} bytes.')
                local.append(target)
            return local[0]


class _RangedPart:
    def __init__(self, handler: StageFileSystemHandler, path: str) -> None:
        self._handler = handler
        self._path = path

    def read_at(self, offset: int, length: int) -> bytes:
        return self._handler._get(self._path, offset, length)

    def close(self) -> None:
        pass


class _DownloadedPart:
    def __init__(self, handler: StageFileSystemHandler, path: str,
                 size: int) -> None:
        self._handler = handler
        self._path = path
        self._size = size

    def read_at(self, offset: int, length: int) -> bytes:
        with open(self._handler._downloaded(self._path, self._size),
                  'rb') as reader:
            reader.seek(offset)
            return reader.read(length)

    def close(self) -> None:
        pass


def stage_filesystem(session: SnowflakeSession,
                     location: str) -> Tuple[PyFileSystem, str]:
    r"""A filesystem for ``@DB.SCHEMA.STAGE/prefix`` and its normalized path."""
    stage, prefix, _ = _split_stage_location(location)
    root = f'{stage}/{prefix}' if prefix else stage
    return PyFileSystem(StageFileSystemHandler(session, root)), root
